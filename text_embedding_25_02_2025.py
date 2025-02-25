
import warnings
import torch
from transformers import pipeline
import fitz  # PyMuPDF for PDF extraction
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from transformers import pipeline

# Download the NLTK punkt tokenizer if not already available.
nltk.download('punkt_tab')

# Initialize the SentenceTransformer model for dense embeddings.
embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)

# Create a knowledge base as a list of dictionaries.
# Each dictionary holds "text" and "page" (None for non-PDF entries).
knowledge_entries = []

# Non-PDF content (page set to None).
original_texts = []

# Add the non-PDF texts to the knowledge base.
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
pdf_path = "PS_2.1_011_1756_01 (2).pdf"  # <-- Update with your PDF file path.
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

# -------------------------------------------------------------------
# NOTE: Initialize your Groq API client before this point.
# For example, if you have a client library for Groq:
# from groq_client import GroqClient
# client = GroqClient(api_key="YOUR_API_KEY")
# -------------------------------------------------------------------
# Suppress the batch_size deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure the correct device (CPU for Ryzen)
device = "cpu"

# Load the text-generation model
pipe = pipeline(
    "text-generation",
    model="google/gemma-2-9b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=device,
)

while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break

    best_entry, score = hybrid_search(query)
    page_info = f"Page: {best_entry['page']}" if best_entry['page'] is not None else "Page: N/A"
    result_text = (
        f"Best Match:\n{best_entry['text']}\n"
        f"{page_info}\n"
        f"(Score: {score:.4f})"
    )
    print("\n--- Search Result ---")
    print(result_text)

    # Retrieve the additional prompt for the Groq API.
    user_prompt = input("Enter additional prompt for Groq API (or press Enter to skip): ").strip()
    one_line = " ".join(line.strip() for line in best_entry['text'].splitlines() if line.strip())

    # Build a combined prompt using both the search result and the user prompt.
    if user_prompt:
        combined_prompt ="User Prompt: Please respond in english {user_prompt}"+one_line
    else:
        combined_prompt = best_entry['text']

    # Feed the combined prompt to the Groq API for text generation.
    try:
        # Call the pipeline with the combined prompt.
        generated = pipe(combined_prompt,max_new_tokens=600)
        # Extract the generated text.
        generated_text = generated[0]['generated_text']
    except Exception as e:
        generated_text = f"Error generating text: {e}"

    print("\n--- Generated Text ---")
    print(generated_text)
    print("\n----------------------\n")

# # Suppress the batch_size deprecation warning
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# # Ensure the correct device (CPU for Ryzen)
# device = "cpu"

# # Load the text-generation model
# pipe = pipeline(
#     "text-generation",
#     model="google/gemma-2-9b-it",
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device=device,
# )
# text = """Projektsteuerung plant und steuert die
# Typgenehmigung / Homologation des Antriebsstrangs
# der Marken Volkswagen, Volkswagen
# Nutzfahrzeuge und anderer Marken gemäß
# abgestimmter Aufgabenteilung für den
# europäischen Markt und UN-ECE-Märkte zur
# Sicherstellung konformer Fahrzeug-SOP und
# Modellpflegen.
# Aufgaben:
# - Fahrzeugprojekte der Homologation zur
# Typgenehmigung planen, steuern und
# realisieren
# - Ergebnisse und Status in Gremien und
# Managementkreisen vertreten und
# präsentieren
# - Problemlösungen im Projekt koordinieren und
# vorantreiben
# - Eingangsgrößen für den Start der
# Homologation Antriebsstrang bewerten und
# sicherstellen
# Aufgabenerweiterung für die ECE-Gruppenbildung:
# - ECE-Gruppenbildung nach festgelegten Kriterien
# durchführen"""

# one_line = " ".join(line.strip() for line in text.splitlines() if line.strip())
# print(one_line)

# # Define multiple user prompts
# prompts = [
#     {"role":"user","content":"Please respond in English for this - "+one_line}
# ]

# # Generate responses for each prompt
# outputs = pipe(prompts, max_new_tokens=1900)

# # Print output for each prompt
# # for i, output in enumerate(outputs):
# #     print(f"Response for prompt {i+1}:")
# #     print(output["generated_text"][-1]["content"].strip())
# #     print("=" * 50)
# # Extract assistant response
# assistant_response = outputs[0]["generated_text"][-1]["content"].strip()

# # Print output
# print(assistant_response)
