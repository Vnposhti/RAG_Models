import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langgraph.graph import Graph  

import os
from dotenv import load_dotenv
load_dotenv('.config')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def preprocessing(input_data):
    pdf_file = input_data['pdf_file']
    reader = PdfReader(pdf_file)
    full_text = "".join(page.extract_text() for page in reader.pages)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_segments = text_splitter.split_text(full_text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_segments, embeddings)
    vector_store.save_local("vectorstore")
    loaded_vector_store = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    return {
            "vector_store": loaded_vector_store,  
            "query": input_data['query']
        }
        
def retrieval_generation(input_data):
    llm = GoogleGenerativeAI(model="gemini-2.0-flash-exp", api_key=GOOGLE_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=input_data['vector_store'].as_retriever())
    response = qa_chain.invoke(input_data['query'])
    return response.get("result")

graph = Graph()
graph.add_node("Preprocess PDF", preprocessing)
graph.add_node("RAG Module", retrieval_generation)
graph.add_edge("Preprocess PDF", "RAG Module")
graph.set_entry_point("Preprocess PDF")
graph.set_finish_point("RAG Module")
app = graph.compile()

st.title("PDF Query Using LangGraph and Google Generative AI")
pdf_file = st.file_uploader("Upload a PDF File", type="pdf")
query = st.text_input("Enter your query", "")
if st.button("Answer"):
    if pdf_file is not None and query:
        input_data = {
            "pdf_file": pdf_file,
            "query": query
        }
        result = app.invoke(input_data)
        if "vector_store" in result:
            input_data['vector_store'] = result['vector_store']
        answer = app.invoke(input_data)
        st.write("Answer:", answer)