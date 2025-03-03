#pip install nltk PyPDF2 pypdfium2 pillow transformers sentence-transformers rank_bm25 langdetect



import os
import warnings
import torch
import numpy as np
import PyPDF2
import pypdfium2 as pdfium
import nltk
from PIL import Image
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import multiprocessing
import pickle
from langdetect import detect, DetectorFactory

# Ensure reproducible language detection
DetectorFactory.seed = 0

# Download the NLTK tokenizer data if not already available.
nltk.download('punkt')

# Load Embedding Model
embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)

# In-memory storage for embeddings and metadata
sentence_embeddings = {}  # {uid: embedding}
sentence_metadata = {}    # {uid: {"text": str, "page": int, "document": str, "language": str}}
document_embeddings = {}  # {doc_name: embedding}

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from each page of the PDF and returns a list of entries with text, page number,
    sentence ID, and detected language.
    """
    reader = PyPDF2.PdfReader(pdf_path)
    page_entries = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:
            sentences = nltk.sent_tokenize(page_text)
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    try:
                        lang = detect(sentence)
                    except Exception:
                        lang = "unknown"
                    page_entries.append({
                        "text": sentence,
                        "page": page_number,
                        "sentence_id": i,
                        "language": lang
                    })
    return page_entries

def store_pdf_embeddings(pdf_path):
    """
    Extracts text, generates embeddings, and stores them in memory.
    """
    doc_name = os.path.basename(pdf_path)
    cache_dir = "embeddings_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{doc_name}.pkl")
    
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        pdf_entries = data["pdf_entries"]
        embeddings = data["embeddings"]
        print(f"Loaded cached embeddings for {doc_name}")
    else:
        pdf_entries = extract_text_from_pdf(pdf_path)
        texts = [entry["text"] for entry in pdf_entries]
        embeddings = embedding_model.encode(texts, batch_size=32).tolist()
        with open(cache_file, "wb") as f:
            pickle.dump({"pdf_entries": pdf_entries, "embeddings": embeddings}, f)
        print(f"Computed and cached embeddings for {doc_name}")
    
    # Debug: print detected languages for each sentence
    for entry in pdf_entries:
        print(f"Sentence: {entry['text'][:50]}... | Detected Language: {entry.get('language', 'unknown')}")
    
    # Store in memory
    for i, entry in enumerate(pdf_entries):
        uid = f"{doc_name}_{entry['page']}_{entry['sentence_id']}"
        sentence_embeddings[uid] = embeddings[i]
        sentence_metadata[uid] = {
            "text": entry["text"],
            "page": entry["page"],
            "document": doc_name,
            "language": entry.get("language", "unknown")
        }
    
    # Compute and store document embedding (mean of sentence embeddings)
    if embeddings:
        doc_embedding = np.mean(embeddings, axis=0).tolist()
        document_embeddings[doc_name] = doc_embedding
        print(f"Stored document embedding for {doc_name}")


def hybrid_search(query, top_k=3, filter_doc=None, context_window=2):
    """
    Performs hybrid search combining dense and sparse retrieval using in-memory storage.
    Incorporates language detection (with fallback) and contextual sentence retrieval,
    along with scoring and ranking functionalities.
    
    The function:
      - Detects the language of the query and filters sentences based on it (allowing language 'unknown').
      - Computes dense distances via query embeddings and retrieves an initial candidate pool.
      - Builds a BM25 model on candidate texts to compute sparse relevance scores.
      - Normalizes both dense and sparse scores and combines them (70% dense, 30% sparse).
      - Extracts surrounding context sentences from the same document/page for each top candidate.
    
    Args:
        query (str): User's search query.
        top_k (int): Number of top results to return.
        filter_doc (str, optional): If specified, filters results by document name.
        context_window (int): Number of sentences before and after the main hit to include as context.
    
    Returns:
        list: A list of dictionaries containing:
              - text: The full text (main sentence plus context).
              - page: The page number.
              - document: The document name.
              - type: The sentence type.
              - score: The final hybrid score.
    """
    # Detect query language
    try:
        query_lang = detect(query)
        print(f"Query detected language: {query_lang}")
    except Exception:
        query_lang = None
        print("Could not detect language for query.")

    # Compute query embedding
    query_embedding = embedding_model.encode([query]).tolist()[0]

    # Helper: search sentences with optional language filtering
    def search_sentences(use_language_filter=True):
        results = []
        for uid, emb in sentence_embeddings.items():
            meta = sentence_metadata[uid]
            if filter_doc and meta["document"] != filter_doc:
                continue
            if use_language_filter and query_lang:
                # Skip sentences that do not match the query language (unless language is 'unknown')
                if meta.get("language") != "unknown" and meta.get("language") != query_lang:
                    continue
            # Compute Euclidean distance as dense retrieval score
            dist = np.linalg.norm(np.array(emb) - np.array(query_embedding))
            results.append((uid, dist))
        return results

    # Run dense retrieval with language filtering
    distances = search_sentences(use_language_filter=True)
    if not distances:
        print("No matching sentences found with language filter. Falling back to search without language filtering.")
        distances = search_sentences(use_language_filter=False)
        if not distances:
            print("No sentences available for search.")
            return []

    # Sort dense retrieval results (lower distance is better)
    distances.sort(key=lambda x: x[1])
    
    # Build candidate pool for BM25 (e.g., twice as many as top_k)
    candidate_pool_size = top_k * 2 if len(distances) >= top_k * 2 else len(distances)
    candidate_ids = [uid for uid, _ in distances[:candidate_pool_size]]
    
    # Prepare texts for BM25
    candidate_texts = [sentence_metadata[uid]["text"] for uid in candidate_ids]
    if not candidate_texts:
        print("No texts retrieved for BM25.")
        return []
    
    tokenized_docs = [word_tokenize(text.lower()) for text in candidate_texts]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = word_tokenize(query.lower())
    sparse_scores = np.array(bm25.get_scores(tokenized_query))
    
    # Prepare dense scores for candidates from pre-computed distances
    candidate_dense_dists = []
    for uid in candidate_ids:
        dist = next(d for (id_d, d) in distances if id_d == uid)
        candidate_dense_dists.append(dist)
    candidate_dense_dists = np.array(candidate_dense_dists)
    
    # Normalize dense scores (convert distances to similarity values in [0,1])
    max_dense = np.max(candidate_dense_dists) if np.max(candidate_dense_dists) > 0 else 1
    dense_scores = 1 - (candidate_dense_dists / max_dense)
    
    # Normalize sparse scores
    max_sparse = np.max(sparse_scores) if np.max(sparse_scores) > 0 else 1
    sparse_scores_norm = sparse_scores / max_sparse
    
    # Combine scores: 70% dense, 30% sparse
    hybrid_scores = 0.7 * dense_scores + 0.3 * sparse_scores_norm
    
    # Re-rank candidates based on combined hybrid score (higher is better)
    sorted_indices = np.argsort(hybrid_scores)[::-1]
    
    best_results = []
    for i in sorted_indices[:top_k]:
        uid = candidate_ids[i]
        meta = sentence_metadata[uid]
        doc_name = meta["document"]
        page = meta.get("page", None)
        # Extract sentence index from uid; assumes format like "document_page_sentenceid"
        try:
            sentence_id = int(uid.split("_")[-1])
        except Exception:
            sentence_id = 0
        
        # Retrieve context sentences from the same document and page
        context_texts = []
        for offset in range(-context_window, context_window + 1):
            neighbor_uid = f"{doc_name}_{page}_{sentence_id + offset}"
            if neighbor_uid in sentence_metadata:
                context_texts.append(sentence_metadata[neighbor_uid]["text"])
        full_text = " ".join(context_texts)
        
        best_results.append({
            "text": full_text,
            "page": page,
            "document": doc_name,
            "type": meta.get("type", "sentence"),
            "score": hybrid_scores[i]
        })
    
    return best_results


# PDF Rendering Functions
def get_total_pages(pdf_path):
    """Return the total number of pages in the PDF using PyPDF2."""
    reader = PyPDF2.PdfReader(pdf_path)
    return len(reader.pages)

def render_page(pdf_path, page_number, scale=1.0, output_path=None):
    """
    Render the specified page using pypdfium2 at the given scale.
    Saves the image if an output path is provided.
    """
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf.get_page(page_number)
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil()
    if output_path:
        pil_image.save(output_path)
        print(f"Saved rendered page {page_number + 1} to: {output_path}")
    page.close()
    pdf.close()
    return pil_image

# def process_specific_page(pdf_path, page_number, scale=7.5, output_dir="rendered_page"):
#     """
#     Render and scale only the specified page of the PDF.
#     """
#     total_pages = get_total_pages(pdf_path)
#     if page_number < 0 or page_number >= total_pages:
#         print(f"Page number {page_number + 1} is out of range. Total pages: {total_pages}")
#         return None
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, f"scaled_page_{page_number + 1}.png")
#     return render_page(pdf_path, page_number, scale=scale, output_path=output_path)

def process_specific_page(pdf_path, page_number, scale=7.5, output_dir="rendered_page"):
    """
    Render and scale only the specified page of the PDF.
    Saves the image with the document name and page number.
    """
    total_pages = get_total_pages(pdf_path)
    if page_number < 0 or page_number >= total_pages:
        print(f"Page number {page_number + 1} is out of range. Total pages: {total_pages}")
        return None
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract document name without extension
    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{doc_name}_page_{page_number + 1}.png")
    
    return render_page(pdf_path, page_number, scale=scale, output_path=output_path)


# Setup the Text-Generation Pipeline
warnings.filterwarnings("ignore", category=DeprecationWarning)
device = "cpu"
pipe = pipeline(
    "text-generation",
    model="google/gemma-2-9b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=device,
)

# Process and Store PDFs
pdf_files = [
    "C:/Users/alukkib/Documents/Hybrid RAG/AutomotiveSPICE_PAM_31.pdf",
    "C:/Users/alukkib/Documents/Hybrid RAG/Automotive_SPICE_PAM_30 1.pdf",
    "C:/Users/alukkib/Documents/Hybrid RAG/PS_1.2_011_1583_017 Software Updates für Fahrzeuge verwalten_V01.01.pdf",
    "C:/Users/alukkib/Documents/Hybrid RAG/PS_2.1_011_1756_01 (2).pdf",
    "C:/Users/alukkib/Documents/Hybrid RAG/PS_2.1_011_1075_05_Relevante Eingangsgrößen für P3+ Anmeldepackage erzeugen.pdf"
]
for pdf_file in pdf_files:
    store_pdf_embeddings(pdf_file)

# Interactive Query Loop
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    
    doc_filter = input("Filter by document name? (Press Enter to skip): ").strip() or None
    best_results = hybrid_search(query, top_k=5, filter_doc=doc_filter)
    if not best_results:
        print("No results found. Please try another query.")
        continue

    print("\n--- Top 5 Search Results ---")
    for res in best_results:
        print(f"Document: {res['document']}, Page {res['page']}")
        print(f"Text: {res['text']}")
        print(f"Score: {res['score']:.4f}")
        print("-" * 50)

    # Optionally, show the rendered page as an image
    show_image = input("Would you like to see the retrieved page as an image? (yes/no): ").strip().lower()
    if show_image == "yes":
        for res in best_results:
            pdf_path = next((path for path in pdf_files if os.path.basename(path) == res["document"]), None)
            if pdf_path:
                process_specific_page(pdf_path, res["page"] - 1, scale=10.0)

    user_prompt = input("Enter additional prompt for text generation (or press Enter to skip): ").strip()
    if user_prompt:
        combined_prompt = f"{user_prompt}\nContext:\n" + "\n".join([res["text"] for res in best_results])
        generated = pipe(combined_prompt, max_new_tokens=900)
        print("\n--- Generated Text ---")
        print(generated[0]['generated_text'])
