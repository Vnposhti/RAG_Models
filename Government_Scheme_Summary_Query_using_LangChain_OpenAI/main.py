import requests
from bs4 import BeautifulSoup
import faiss
import os
import openai
from dotenv import load_dotenv
import pdfplumber
from io import BytesIO
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import numpy as np
import streamlit as st

# Load environment variables
load_dotenv(".config")  # Load the environment variables from the .config file
openai_api_key = os.getenv("OPENAI_API_KEY")

# Fetch content
def fetch_text_content(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        # Check if the content is a PDF
        if 'application/pdf' in response.headers.get('Content-Type', ''):
            with pdfplumber.open(BytesIO(response.content)) as pdf:
                text = ''.join(page.extract_text() for page in pdf.pages)
            return text
        else:
            # If it's not a PDF, treat it as a webpage
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup.get_text()  # Extract all text from the HTML
    else:
        return f"Failed to fetch the content from URL: {url}, Status Code: {response.status_code}"
    
# Preprocess text into chunks
def preprocess_text(text):
    # Split text into chunks of size 500 with overlap of 50 characters
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)
    
# Store both text chunks and associated metadata
def store_text_and_metadata(text_chunks, url):
    metadata = []
    for i, chunk in enumerate(text_chunks):
        metadata_entry  = {
            "chunk_id": i,
            "source": url,
            "page": i // 10 + 1,  # Example page number calculation
            "text": chunk  # Add the text to the metadata entry
        }
        metadata.append(metadata_entry)
    return metadata

# Generating a FAISS index
def generate_faiss_index(text_chunks_with_metadata):
    # Extract the text from the chunk+metadata structure for FAISS processing
    texts = [entry["text"] for entry in text_chunks_with_metadata]
    
    # Generate embeddings using OpenAI's Embeddings model
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)
    
    # Create the FAISS index
    faiss_index = FAISS.from_texts(texts, embeddings)
    
    # Store metadata separately
    metadata = {i: entry for i, entry in enumerate(text_chunks_with_metadata)}
    
    return faiss_index, metadata

# Save FAISS index using FAISS's write_index
def save_faiss_index(faiss_index, metadata, file_path, metadata_file_path):
    # Save the FAISS index to disk
    faiss.write_index(faiss_index.index, file_path)
    # Save metadata using pickle
    with open(metadata_file_path, "wb") as f:   
        pickle.dump(metadata, f)     

# Load FAISS index using FAISS's read_index
def load_faiss_index(file_path, metadata_file_path):
    # Load the FAISS index from disk
    index = faiss.read_index(file_path)
    # Load metadata using pickle
    with open(metadata_file_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata
      
# Initialize the OpenAI client with your API key
llm = OpenAI(model="gpt-4o", api_key=openai_api_key)

# Summarization
def summarize_text(text):
    prompts = {
        "benefits": "Extract and summarize the benefits of the scheme:",
        "application_process": "Describe the application process for the scheme:",
        "eligibility": "Who is eligible for the scheme:",
        "documents": "List the documents required for the scheme:",
    }
    summaries = {}
    for key, prompt in prompts.items():
        response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": f"{prompt}\n{text}"}
                ])
        summaries[key] = response.choices[0].message.content
    return summaries

# Query System 
def query_system(query, faiss_index, metadata):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)
    query_embedding = embeddings.embed_query(query)
    
    # Convert the query embedding to a numpy array and reshape it
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    # Perform the search
    distances, indices = faiss_index.search(query_embedding, k=1)
    
    # Retrieve the similar texts based on the indices
    similar_texts = [metadata[idx]["text"] for idx in indices[0]]
    
    # Iterate over each text and generate a response
    answers = []
    for text in similar_texts:
        # Create a message for the chat model
        response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": text}
                ]
            )
        # Extract the content of the response
        answers.append(response.choices[0].message.content)   
       
    # Extract source information for each text
    sources = [metadata[idx]["source"] for idx in indices[0]]
    return answers, sources

# Main method for Streamlit app
def main():
    st.title("Scheme Research Application")

    # Step 1: Fetch content from URLs
    st.sidebar.header("Input URLs")
    urls = st.sidebar.text_area("Enter URLs (one per line):", height=200)

    if st.sidebar.button("Process URLs"):
        if urls.strip():
            url_list = urls.split("\n")
            all_text_chunks = []
            all_metadata = []

            with st.spinner("Processing..."):
                for url in url_list:
                    text = fetch_text_content(url)

                    if text.startswith("Failed to fetch"):
                        st.sidebar.error(text)
                        continue

                    text_chunks = preprocess_text(text)
                    metadata = store_text_and_metadata(text_chunks, url)

                    all_text_chunks.extend(text_chunks)
                    all_metadata.extend(metadata)

                if all_text_chunks:
                    faiss_index, metadata = generate_faiss_index(all_metadata)

                    # Save index and metadata for later use
                    save_faiss_index(
                        faiss_index,
                        metadata,
                        "faiss_index.idx",
                        "metadata.pkl"
                    )

    # Step 2: Display summaries
    st.header("Summary")
    
    for url in urls.split("\n"):
        content = fetch_text_content(url)
        summaries = summarize_text(content)

        for key, summary in summaries.items():
            st.write(f"**{key.capitalize()}:** {summary}")

    # Step 3: Query the system
    st.header("Ask me")
    query = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        try:
            # Load the FAISS index and metadata
            faiss_index, metadata = load_faiss_index("faiss_index.idx", "metadata.pkl")

            if query.strip():
                answers, sources = query_system(query, faiss_index, metadata)

                for answer, source in zip(answers, sources):
                    st.write(f"### Answer")
                    st.write(answer)
                    st.write(f"**Source URL:** {source}")
            else:
                st.error("Please enter a valid query.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
