import os
import warnings
import torch
from transformers import pipeline
import PyPDF2
import pypdfium2 as pdfium
from PIL import Image
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import io
 
# Download the NLTK tokenizer data if not already available.
nltk.download('punkt_tab')
 
# --------------------------
# PDF Text Extraction with PyPDF2
# --------------------------
def extract_text_from_pdf(pdf_path):
    """Extracts text from each page of the PDF and returns a list of entries with text and page number."""
    reader = PyPDF2.PdfReader(pdf_path)
    page_entries = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:  # Ensure there is text on the page
            sentences = nltk.sent_tokenize(page_text)
            for sentence in sentences:
                if sentence.strip():
                    page_entries.append({"text": sentence, "page": page_number})
    return page_entries
 
# --------------------------
# Image Rendering Functions using pdfium2
# --------------------------
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
    # Render the page at a high resolution by applying the scale.
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil()
    if output_path:
        pil_image.save(output_path)
        print(f"Saved rendered page {page_number + 1} with image to: {output_path}")
    page.close()
    pdf.close()
    return pil_image
 
def process_specific_page(pdf_path, page_number, scale=1.0, output_dir="rendered_page"):
    """
    Render and scale only the specified page of the PDF.
    This function simply renders the given page, scales it, and saves the result.
    """
    total_pages = get_total_pages(pdf_path)
    if page_number < 0 or page_number >= total_pages:
        print(f"Page number {page_number + 1} is out of range. Total pages: {total_pages}")
        return
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"scaled_page_{page_number + 1}.png")
    render_page(pdf_path, page_number, scale=scale, output_path=output_path)
 
# --------------------------
# Setup for Retrieval and Text Generation
# --------------------------
# Path to your PDF file.
pdf_path = "C:/Users/alukkib/Documents/Hybrid RAG/AutomotiveSPICE_PAM_31.pdf"  # <-- Update with your PDF file path.
 
# Create a knowledge base.
knowledge_entries = []
try:
    pdf_entries = extract_text_from_pdf(pdf_path)
    knowledge_entries.extend(pdf_entries)
    print("PDF content successfully integrated into the knowledge base!")
except Exception as e:
    print(f"Error reading PDF: {e}")
 
# Prepare texts and compute embeddings.
texts = [entry["text"] for entry in knowledge_entries]
embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
dense_embeddings = embedding_model.encode(texts)
 
# Prepare BM25 indexing.
tokenized_docs = [word_tokenize(doc.lower()) for doc in texts]
bm25 = BM25Okapi(tokenized_docs)
 
def hybrid_search(query):
    """Performs a hybrid search combining dense and sparse retrieval.
       Returns the best matching entry (with text and page) and its score."""
    # Dense retrieval.
    query_embedding = embedding_model.encode(query)
    dense_scores = np.dot(dense_embeddings, query_embedding)
    dense_scores = dense_scores / np.max(dense_scores)  # Normalize dense scores.
 
    # Sparse retrieval.
    tokenized_query = word_tokenize(query.lower())
    sparse_scores = np.array(bm25.get_scores(tokenized_query))
    max_sparse = np.max(sparse_scores)
    if max_sparse > 0:
        sparse_scores = sparse_scores / max_sparse  # Normalize sparse scores.
    else:
        sparse_scores = np.zeros_like(sparse_scores)
 
    # Combine the scores with equal weighting.
    hybrid_scores = 0.5 * dense_scores + 0.5 * sparse_scores
    best_match_index = np.argmax(hybrid_scores)
    best_entry = knowledge_entries[best_match_index]
    return best_entry, hybrid_scores[best_match_index]
 
# Setup the text-generation pipeline.
warnings.filterwarnings("ignore", category=DeprecationWarning)
device = "cpu"
pipe = pipeline(
    "text-generation",
    model="google/gemma-2-9b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=device,
)
 
# --------------------------
# Interactive Query Loop
# --------------------------
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break
 
    # Retrieve the best matching entry.
    best_entry, score = hybrid_search(query)
    page_info = f"Page: {best_entry['page']}" if best_entry['page'] is not None else "Page: N/A"
    result_text = (
        f"Best Match:\n{best_entry['text']}\n"
        f"{page_info}\n"
        f"(Score: {score:.4f})"
    )
    print("\n--- Search Result ---")
    print(result_text)
 
    # Ask if the user wants to see an image of the retrieved page.
    if best_entry['page'] is not None:
        want_image = input("Would you like to render an image of this page? (yes/no): ").strip().lower()
        if want_image == "yes":
            # Adjust the page number to be zero-indexed.
            retrieved_page_index = best_entry['page'] - 1
            # Set your desired scale; adjust if needed.
            process_specific_page(pdf_path, retrieved_page_index, scale=7.5, output_dir="relevante")
 
    # Retrieve additional prompt for text generation.
    user_prompt = input("Enter additional prompt for text generation (or press Enter to skip): ").strip()
    one_line = " ".join(line.strip() for line in best_entry['text'].splitlines() if line.strip())
    if user_prompt:
        combined_prompt = "User Prompt: Please respond in English\n" + user_prompt + " " + one_line
    else:
        combined_prompt = best_entry['text']
 
    # Generate text using the pipeline.
    try:
        generated = pipe(combined_prompt, max_new_tokens=900)
        generated_text = generated[0]['generated_text']
    except Exception as e:
        generated_text = f"Error generating text: {e}"
 
    print("\n--- Generated Text ---")
    print(generated_text)
    print("\n----------------------\n")
