from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import PyPDF2
import streamlit as st

import os
from dotenv import load_dotenv
load_dotenv('.config')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def extract_text_from_pdf(uploaded_pdf):
    pdf_content = ""
    reader = PyPDF2.PdfReader(uploaded_pdf)
    for page in reader.pages:
        pdf_content += page.extract_text() + "\n"
    return pdf_content

def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    texts = text_splitter.split_text(text)
    return texts

def create_faiss_index_with_langchain(texts):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",api_key=GOOGLE_API_KEY)
    docsearch = FAISS.from_texts(texts, embeddings)
    return docsearch

def save_faiss_index(docsearch, directory='faiss_index'):
    docsearch.save_local(directory)

def load_faiss_index(directory='faiss_index'):
    docsearch = FAISS.load_local(directory)
    return docsearch

def answer_question_with_langchain(query, docsearch):
    llm = GoogleGenerativeAI(model="gemini-2.0-flash-exp",api_key=GOOGLE_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=docsearch.as_retriever())
    answer = qa_chain.invoke(query)
    return answer.get("result", "No result found.")

def main():
    st.title("PDF to FAISS Query System using LangChain and Google Generative AI")

    st.write("This app uses LangChain and Google GenAI to extract text from a PDF, chunk the text, generate embeddings, index them using FAISS, and answer questions based on the PDF's content.")

    st.write("Please upload a PDF file to get started.")

    uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_pdf is not None:
        with st.spinner("Processing..."):
            pdf_content = extract_text_from_pdf(uploaded_pdf)
            texts = chunk_text(pdf_content)
            docsearch = create_faiss_index_with_langchain(texts)
            save_faiss_index(docsearch)
            query = st.text_input("Ask a question to get an answer based on the text in the PDF:")
            if st.button("Get Answer"):
                answer = answer_question_with_langchain(query, docsearch)
                st.write("### Answer:")
                st.write(answer)


if __name__ == "__main__":
    main()