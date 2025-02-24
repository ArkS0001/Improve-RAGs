import fitz  # PyMuPDF for PDF extraction
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

# Download the NLTK punkt tokenizer if not already available.
nltk.download('punkt')

# Initialize the SentenceTransformer model for dense embeddings.
embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)

# Create a knowledge base as a list of dictionaries.
# Each dictionary holds "text" and "page" (None for non-PDF entries).
knowledge_entries = []

# Non-PDF content (page set to None).
original_texts = []

# Add original texts to the knowledge base.
for text in original_texts:
    knowledge_entries.append({"text": text, "page": None})

# Function to extract PDF text page-by-page and split it into sentences.
def extract_text_from_pdf(pdf_path):
    """Extracts text from each page of the PDF and returns a list of entries with text and page number."""
    doc = fitz.open(pdf_path)
    page_entries = []
    for page_number, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        sentences = nltk.sent_tokenize(page_text)
        for sentence in sentences:
            if sentence.strip():
                page_entries.append({"text": sentence, "page": page_number})
    return page_entries

# Load PDF content and add its entries (with page numbers) to the knowledge base.
pdf_path = "/content/PS_2.1_011_1756_01 (2).pdf"  # <-- Update with your PDF file path.
try:
    pdf_entries = extract_text_from_pdf(pdf_path)
    knowledge_entries.extend(pdf_entries)
    print("PDF content successfully integrated into the knowledge base!")
except Exception as e:
    print(f"Error reading PDF: {e}")

# Prepare a list of texts for embedding and BM25 indexing.
texts = [entry["text"] for entry in knowledge_entries]

# Compute dense embeddings for all texts.
dense_embeddings = embedding_model.encode(texts)

# Tokenize texts for BM25 sparse retrieval.
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

# Interactive query loop: prompt user input and print full outputs.
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    best_entry, score = hybrid_search(query)
    page_info = f"Page: {best_entry['page']}" if best_entry['page'] is not None else "Page: N/A"
    print("\n--- Search Result ---")
    print(f"Best Match: {best_entry['text']}")
    print(page_info)
    print(f"(Score: {score:.4f})\n")
