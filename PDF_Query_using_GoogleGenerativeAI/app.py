import google.generativeai as genai
import os
from dotenv import load_dotenv
import PyPDF2
import faiss
import numpy as np
import pickle
import streamlit as st

load_dotenv('.config')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def extract_text_from_pdf(uploaded_pdf):
    text = ""
    reader = PyPDF2.PdfReader(uploaded_pdf)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500):
    texts= [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return texts

def generate_embeddings(texts, model="models/text-embedding-004"):
    embeddings = []
    for text in texts:
        embedding = genai.embed_content(model=model, content=text)
        embeddings.append(embedding)
    return embeddings

def create_faiss_index(texts,embeddings):
    embedding_dim = len(embeddings[0]['embedding'])
    index = faiss.IndexFlatL2(embedding_dim)
    # Add embeddings and texts to the index
    metadata = []  
    for text, embedding in zip(texts, embeddings):
        index.add(np.array(embedding['embedding'], dtype=np.float32).reshape(1, -1))
        metadata.append(text)
    return index, metadata

# Function to save FAISS index and metadata
def save_faiss_index(index, metadata, index_file="faiss.index", metadata_file="metadata.pkl"):
    faiss.write_index(index, index_file)
    with open(metadata_file, "wb") as file:
        pickle.dump(metadata, file)

def load_faiss_index(index_file="faiss.index", metadata_file="metadata.pkl"):
    index = faiss.read_index(index_file)
    with open(metadata_file, "rb") as file:
        metadata = pickle.load(file)
    return index, metadata

def query_vector_database(query, loaded_index, loaded_metadata):
    query_embedding = genai.embed_content(
        model="models/text-embedding-004",
        content=query
    )['embedding']

    distances, indices = loaded_index.search(np.array(query_embedding, dtype=np.float32).reshape(1, -1), k=5)
    results = [(loaded_metadata[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results

def answer_question(query, loaded_index, loaded_metadata):
    relevant_texts = query_vector_database(query, loaded_index, loaded_metadata)
    context = "\n".join([text for text, _ in relevant_texts])
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    response = model.generate_content(
        f"Answer the question based on the following information:\n{context}\n\nQuestion: {query}"
    )
    return response.candidates[0].content.parts[0].text

def main():
    st.title("PDF to FAISS Query System using Google AI Studio")
    st.write("This app uses the Google AI Studio platform to extract text from a PDF, chunk the text, embed the chunks using a text embedding model, and index the embeddings using FAISS. It then uses the index to answer questions based on the text in the PDF.")
    st.write("Please upload a PDF file to get started.")

    uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_pdf is not None:
        with st.spinner("Processing..."):
            text = extract_text_from_pdf(uploaded_pdf)
            texts = chunk_text(text)
            embeddings = generate_embeddings(texts)
            index, metadata = create_faiss_index(texts, embeddings)  # Pass texts here
            save_faiss_index(index, metadata)
            loaded_index, loaded_metadata = load_faiss_index()
        query = st.text_input("Ask a question to get an answer based on the text in the PDF:")
        if st.button("Answer"):
            answer = answer_question(query, loaded_index, loaded_metadata)
            st.write("### Answer:")
            st.write(answer)

if __name__ == "__main__":
    main()