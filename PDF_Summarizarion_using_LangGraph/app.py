import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langgraph.graph import Graph

import os
from dotenv import load_dotenv
load_dotenv('.config')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def pdf_preprocessing(uploaded_pdf):
    pdf_content = ""
    reader = PdfReader(uploaded_pdf)
    for page in reader.pages:
        pdf_content += page.extract_text() + "\n"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    texts = text_splitter.split_text(pdf_content)
    return texts

def summarize_text(text):
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    prompt = f"Extract Key Features and summarize the headingwise and sub-headingwise main content of the PDF:\n{text}"
    response = model.generate_content(prompt)
    return response.candidates[0].content.parts[0].text

graph = Graph()
graph.add_node("PDF PREPROCESSING", pdf_preprocessing)
graph.add_node("TEXT SUMMARIZATION", summarize_text)
graph.add_edge("PDF PREPROCESSING", "TEXT SUMMARIZATION")
graph.set_entry_point("PDF PREPROCESSING")
graph.set_finish_point("TEXT SUMMARIZATION")
app=graph.compile()

st.title("PDF Summarization with LangGraph and Google Generative AI")
st.write("Upload a PDF file to Summarize")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    st.session_state["uploaded_file"] = uploaded_file
    with st.spinner("Extracting and summarizing content..."):
        result = app.invoke(uploaded_file)
        st.text_area("PDF Summary", value=result, height=1000)