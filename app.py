import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Simple RAG Bot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Simple RAG Chatbot")
st.write("Ask questions about the company policies.")

# -----------------------------
# Knowledge Base
# -----------------------------
documents = [

"""
Employees receive 12 casual leaves annually.
""",

"""
Medical insurance covers spouse and children.
""",

"""
Employees may work remotely 2 days per week.
""",

"""
Sick leave entitlement is 10 days annually.
""",

"""
Travel expenses require manager approval.
"""
]

# -----------------------------
# Load Embedding Model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# Create document embeddings
doc_embeddings = embedding_model.encode(documents)

# -----------------------------
# Load Generator Model
# -----------------------------
@st.cache_resource
def load_generator():
    return pipeline(
        "text-generation",
        model="gpt2"
    )

generator = load_generator()

# -----------------------------
# User Input
# -----------------------------
query = st.text_input(
    "Enter your question:",
    placeholder="Example: How many sick leaves are allowed?"
)

# -----------------------------
# RAG Pipeline
# -----------------------------
if st.button("Get Answer"):

    if query.strip() == "":
        st.warning("Please enter a question.")
    else:

        # Query Embedding
        query_embedding = embedding_model.encode([query])

        # Similarity Search
        scores = cosine_similarity(
            query_embedding,
            doc_embeddings
        )

        best_index = np.argmax(scores)

        context = documents[best_index]

        st.subheader("Retrieved Context")
        st.info(context)

        # Prompt
        prompt = f"""
Answer only from the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

        # Generate Response
        result = generator(
            prompt,
            max_new_tokens=50,
            do_sample=False
        )

        answer = result[0]["generated_text"]

        st.subheader("Generated Answer")
        st.success(answer)