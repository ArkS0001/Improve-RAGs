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

# Download the NLTK tokenizer data if not already available.
nltk.download('punkt')

# Load Embedding Model
embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)

# In-memory storage for embeddings and metadata
sentence_embeddings = {}  # {id: embedding}
sentence_metadata = {}    # {id: {"text": str, "page": int, "document": str}}
document_embeddings = {}  # {doc_name: embedding}

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from each page of the PDF and returns a list of entries with text, page number, and sentence ID.
    """
    reader = PyPDF2.PdfReader(pdf_path)
    page_entries = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:
            sentences = nltk.sent_tokenize(page_text)
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    page_entries.append({
                        "text": sentence,
                        "page": page_number,
                        "sentence_id": i
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
    
    # Store in memory
    for i, entry in enumerate(pdf_entries):
        id = f"{doc_name}_{entry['page']}_{entry['sentence_id']}"
        sentence_embeddings[id] = embeddings[i]
        sentence_metadata[id] = {"text": entry["text"], "page": entry["page"], "document": doc_name}
    
    # Compute and store document embedding
    if embeddings:
        doc_embedding = np.mean(embeddings, axis=0).tolist()
        document_embeddings[doc_name] = doc_embedding
        print(f"Stored document embedding for {doc_name}")

# def hybrid_search(query, top_k=3, filter_doc=None):
#     """
#     Performs hybrid search using in-memory storage.
#     """
#     query_embedding = embedding_model.encode([query]).tolist()[0]
    
#     # Dense retrieval: compute distances to all sentence embeddings
#     distances = []
#     for id, emb in sentence_embeddings.items():
#         if filter_doc and sentence_metadata[id]["document"] != filter_doc:
#             continue
#         dist = np.linalg.norm(np.array(emb) - np.array(query_embedding))
#         distances.append((id, dist))
    
#     # Sort by distance (lower is better)
#     distances.sort(key=lambda x: x[1])
#     top_ids = [id for id, _ in distances[:top_k]]
    
#     best_results = []
#     for id in top_ids:
#         meta = sentence_metadata[id]
#         best_results.append({
#             "text": meta["text"],
#             "page": meta["page"],
#             "document": meta["document"],
#             "type": "sentence",
#             "score": 1.0 / (distances[top_ids.index(id)][1] + 1e-10)  # Inverse distance as score
#         })
    
#     return best_results

def hybrid_search(query, top_k=3, filter_doc=None, context_window=2):
    """
    Performs hybrid search using in-memory storage, retrieving additional surrounding sentences for context.
    
    Args:
        query (str): User's search query.
        top_k (int): Number of top results to return.
        filter_doc (str, optional): If specified, filters results by document name.
        context_window (int): Number of sentences before and after the main hit to include as context.
    
    Returns:
        list: A list of dictionaries containing text, page number, document name, and retrieval score.
    """
    query_embedding = embedding_model.encode([query]).tolist()[0]

    # Compute dense distances for all sentence embeddings
    distances = []
    for id, emb in sentence_embeddings.items():
        if filter_doc and sentence_metadata[id]["document"] != filter_doc:
            continue
        dist = np.linalg.norm(np.array(emb) - np.array(query_embedding))
        distances.append((id, dist))

    # Sort by distance (lower is better)
    distances.sort(key=lambda x: x[1])
    top_ids = [id for id, _ in distances[:top_k]]

    best_results = []
    for id in top_ids:
        meta = sentence_metadata[id]
        doc_name = meta["document"]
        page = meta["page"]
        sentence_id = int(id.split("_")[-1])  # Extract sentence ID

        # Retrieve additional sentences from the same page
        context_text = []
        for offset in range(-context_window, context_window + 1):  # Before & After
            neighbor_id = f"{doc_name}_{page}_{sentence_id + offset}"
            if neighbor_id in sentence_metadata:
                context_text.append(sentence_metadata[neighbor_id]["text"])

        full_text = " ".join(context_text)

        best_results.append({
            "text": full_text,
            "page": page,
            "document": doc_name,
            "type": "sentence",
            "score": 1.0 / (distances[top_ids.index(id)][1] + 1e-10)  # Inverse distance as score
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

def process_specific_page(pdf_path, page_number, scale=7.5, output_dir="rendered_page"):
    """
    Render and scale only the specified page of the PDF.
    """
    total_pages = get_total_pages(pdf_path)
    if page_number < 0 or page_number >= total_pages:
        print(f"Page number {page_number + 1} is out of range. Total pages: {total_pages}")
        return None
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"scaled_page_{page_number + 1}.png")
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

    print("\n--- Top 3 Search Results ---")
    for res in best_results:
        print(f"Document: {res['document']}, Page {res['page']}")
        print(f"Text: {res['text']}")
        print(f"Score: {res['score']:.4f}")
        print("-" * 50)

    # Ask user if they want to see an image of the retrieved page
    show_image = input("Would you like to see the retrieved page as an image? (yes/no): ").strip().lower()
    if show_image == "yes":
        for res in best_results:
            pdf_path = next((path for path in pdf_files if os.path.basename(path) == res["document"]), None)
            if pdf_path:
                process_specific_page(pdf_path, res["page"] - 1, scale=7.5)

    user_prompt = input("Enter additional prompt for text generation (or press Enter to skip): ").strip()
    if user_prompt:
        combined_prompt = f"{user_prompt}\nContext:\n" + "\n".join([res["text"] for res in best_results])
        generated = pipe(combined_prompt, max_new_tokens=900)
        print("\n--- Generated Text ---")
        print(generated[0]['generated_text'])
