# app.py

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Semantic FAQ Assistant",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# TITLE
# -----------------------------
st.title("🤖 Semantic FAQ Assistant")
st.write("Ask questions related to AI, ML, Deep Learning, and Python.")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -----------------------------
# FAQ DATA
# -----------------------------
faq_questions = [
    "What is AI?",
    "What is Machine Learning?",
    "How does Deep Learning work?",
    "What is Python used for?"
]

faq_answers = [
    "AI enables machines to mimic human intelligence.",
    "Machine Learning allows systems to learn from data.",
    "Deep Learning uses neural networks with many layers.",
    "Python is widely used in AI, ML, and web development."
]

# -----------------------------
# GENERATE FAQ EMBEDDINGS
# -----------------------------
faq_embeddings = model.encode(faq_questions)

# -----------------------------
# USER INPUT
# -----------------------------
query = st.text_input(
    "Enter your question:",
    placeholder="Example: How is AI used?"
)

# -----------------------------
# PROCESS QUERY
# -----------------------------
if st.button("Get Answer"):

    if query.strip() == "":
        st.warning("Please enter a question.")
    
    else:

        # Query embedding
        query_embedding = model.encode([query])

        # Similarity scores
        scores = cosine_similarity(
            query_embedding,
            faq_embeddings
        )

        # Best match
        best_index = np.argmax(scores)

        best_question = faq_questions[best_index]
        best_answer = faq_answers[best_index]
        similarity_score = scores[0][best_index]

        # -----------------------------
        # OUTPUT
        # -----------------------------
        st.success("Most Relevant FAQ Found!")

        st.subheader("📌 Matched Question")
        st.write(best_question)

        st.subheader("💡 Answer")
        st.write(best_answer)

        st.subheader("📊 Similarity Score")
        st.write(round(float(similarity_score), 2))

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📚 Available FAQs")

for q in faq_questions:
    st.sidebar.write("✅ " + q)

st.sidebar.markdown("---")
st.sidebar.write("Built using:")
st.sidebar.write("- Streamlit")
st.sidebar.write("- Sentence Transformers")
st.sidebar.write("- Cosine Similarity")