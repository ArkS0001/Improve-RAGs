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
original_texts = [
    # Science & Technology
    "Quantum entanglement describes the phenomenon where two particles remain connected, sharing states instantaneously.",
    "The theory of general relativity explains gravity as the curvature of spacetime caused by mass and energy.",
    "Neural networks, inspired by the human brain, are fundamental to deep learning models used in AI.",
    "Blockchain is a decentralized ledger technology that ensures transparency and security in transactions.",
    "CRISPR technology allows for precise genetic modifications, revolutionizing biotechnology and medicine.",
    "The Turing Test evaluates a machine’s ability to exhibit intelligent behavior indistinguishable from a human.",
    "Fusion energy, the process powering the Sun, has the potential to provide unlimited clean energy on Earth.",
    "Dark matter is an unknown form of matter that does not emit light but exerts gravitational effects in the universe.",
    "Photosynthesis is the process by which plants convert light energy into chemical energy, releasing oxygen.",
    "RNA vaccines, like those for COVID-19, use messenger RNA to instruct cells to build an immune response.",
    "Artificial Intelligence is transforming industries through automation, predictive analytics, and data-driven decision-making.",
    "Superconductors allow electricity to flow without resistance at extremely low temperatures.",
    "Graphene is a one-atom-thick layer of carbon that has extraordinary electrical, thermal, and mechanical properties.",
    "Edge computing reduces latency by processing data closer to the source rather than relying on centralized cloud servers.",

    # Space & Astronomy
    "The James Webb Space Telescope is designed to observe the universe’s first galaxies and exoplanets in infrared light.",
    "Black holes have such a strong gravitational pull that nothing, not even light, can escape from them.",
    "Exoplanets are planets outside our solar system, with thousands discovered using the Kepler and TESS telescopes.",
    "Astrobiology is the study of life’s potential in the universe, searching for microbial life on Mars and Europa.",
    "SpaceX successfully launched the Falcon Heavy, a reusable rocket capable of carrying heavy payloads to deep space.",

    # History & Geography
    "The Great Wall of China was built over several centuries to protect against invasions from northern tribes.",
    "The Industrial Revolution marked a period of rapid technological advancements and urbanization in the 18th century.",
    "The Cold War was a period of geopolitical tension between the United States and the Soviet Union from 1947 to 1991.",
    "Ancient Egyptian civilization is known for its pyramids, hieroglyphic writing, and advances in engineering and astronomy.",
    "The Roman Empire was one of history’s most powerful civilizations, known for its roads, architecture, and military strength.",
    "The Silk Road was an ancient trade network connecting China, India, the Middle East, and Europe.",
    "World War II lasted from 1939 to 1945 and involved major global powers, ending with the defeat of Nazi Germany and Japan.",
    "The Renaissance was a cultural movement in Europe that led to advancements in art, science, and literature.",

    # Current Affairs & Economics
    "Cryptocurrencies like Bitcoin and Ethereum use blockchain technology for decentralized transactions.",
    "Inflation occurs when the prices of goods and services rise, reducing purchasing power over time.",
    "Renewable energy sources such as solar, wind, and hydroelectric power are key to reducing carbon emissions.",
    "The global supply chain crisis has led to increased shipping costs and delays due to pandemic-related disruptions.",
    "Electric vehicles (EVs) are becoming more popular as battery technology improves and charging infrastructure expands.",
    "Artificial Intelligence is increasingly being used in finance for fraud detection and automated trading strategies.",
    "The semiconductor chip shortage has affected industries from automotive to consumer electronics worldwide.",
    "E-commerce has experienced rapid growth, with companies like Amazon and Alibaba dominating the market.",

    # Public Knowledge & Daily Life
    "A balanced diet consists of carbohydrates, proteins, fats, vitamins, and minerals for overall health.",
    "Water makes up about 60% of the human body and is essential for survival.",
    "Sleep is crucial for brain function, memory consolidation, and overall physical health.",
    "Exercise helps reduce the risk of chronic diseases like obesity, heart disease, and diabetes.",
    "Vitamin D is essential for bone health and is obtained from sunlight exposure and certain foods.",
    "Recycling helps reduce waste and conserves natural resources by repurposing materials.",
    "The average human heart beats about 100,000 times per day, pumping blood throughout the body.",
    "Plastic pollution is a major environmental issue, affecting marine life and ecosystems.",
    "Social media platforms influence public opinion and have become a key tool for communication and business.",
    "Artificial Intelligence is being used in customer service through chatbots and virtual assistants.",

    # Cybersecurity & Privacy
    "Multi-factor authentication (MFA) enhances security by requiring multiple forms of verification.",
    "Phishing attacks trick users into providing personal information through deceptive emails or websites.",
    "End-to-end encryption ensures that only the sender and recipient can read a message, protecting privacy.",
    "The Tor network allows anonymous browsing by routing traffic through a series of encrypted relays.",
    "Ransomware attacks encrypt victims’ files and demand payment for their release.",
    "Cybersecurity best practices include using strong passwords, updating software, and avoiding suspicious links.",
    "Data breaches can expose sensitive personal information, leading to identity theft and financial fraud.",

    # Medicine & Biology
    "The human genome contains approximately 20,000-25,000 genes that determine physical traits and functions.",
    "Antibiotics help fight bacterial infections but are ineffective against viruses like the common cold.",
    "Stem cells have the potential to develop into different types of cells, offering possibilities for regenerative medicine.",
    "Cancer is a group of diseases characterized by uncontrolled cell growth, which can spread to other body parts.",
    "The brain’s hippocampus is critical for memory formation and learning.",
    "Dopamine is a neurotransmitter that plays a key role in pleasure, motivation, and reward.",
    "CRISPR-Cas9 is a revolutionary gene-editing technology allowing precise modifications of DNA.",
    "Alzheimer’s disease is a neurodegenerative condition that affects memory and cognitive function.",
    "The placebo effect occurs when a patient experiences improvements despite receiving a non-active treatment.",

    # Emerging Technologies
    "5G technology offers higher data speeds and lower latency compared to previous generations of mobile networks.",
    "Quantum computing has the potential to solve complex problems much faster than classical computers.",
    "Autonomous vehicles use sensors and AI to navigate without human intervention.",
    "Nanotechnology involves manipulating materials at the atomic and molecular scale for applications in medicine and industry.",
    "Smart homes use IoT devices to automate lighting, heating, and security systems.",
    "Biometric authentication, such as facial recognition and fingerprint scanning, enhances digital security.",
    "Augmented reality (AR) overlays digital content onto the real world, enhancing user experiences in gaming and retail.",
    "The metaverse is a virtual world where users can interact through digital avatars in immersive environments.",
    "AI-powered drug discovery accelerates the development of new pharmaceuticals by predicting molecule interactions.",

    # Environment & Sustainability
    "Climate change results from human activities like deforestation and burning fossil fuels, leading to global warming.",
    "Carbon capture technology aims to reduce greenhouse gas emissions by storing carbon underground.",
    "Deforestation contributes to biodiversity loss and disrupts ecosystems.",
    "Ocean acidification, caused by increased CO2 levels, harms marine life, including coral reefs.",
    "The Paris Agreement is an international treaty aimed at reducing global carbon emissions to combat climate change.",
    "Biodiversity is essential for ecosystem stability, providing resources like food, medicine, and clean air."
]

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
