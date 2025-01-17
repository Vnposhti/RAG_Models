### PDF to FAISS Query System using LangChain and Google Generative AI
This repository demonstrates a PDF-based Retrieval-Augmented Generation (RAG) system using LangChain and Google Generative AI. It enables users to upload a PDF document, process its content, create embeddings using Google GenAI, index the embeddings in FAISS, and answer user queries based on the content of the PDF.

### 🚀 Features
- Extracts text from PDFs using PyPDF2.
- Splits text into manageable chunks using LangChain's RecursiveCharacterTextSplitter.
- Generates text embeddings using Google Generative AI's embedding-001 model.
- Stores embeddings in a FAISS vector database for efficient retrieval.
- Retrieves relevant content using LangChain's retrieval mechanism.
- Generates answers to user queries with Google GenAI's gemini-2.0-flash-exp model.
- Provides a simple and interactive user interface built with Streamlit.

### Live Demo
You can access the live Streamlit application at the following link: [PDF Query System using LangChain and Google Generative AI](https://ragmodels-langchain-goooglegenai-pdfquery.streamlit.app/)

### 💡 Usage
**Upload a PDF:**
Use the file uploader to upload a PDF document containing the information you want to process.

**Process the PDF:**
The app extracts and chunks the text from the PDF.
Embeddings for the text chunks are generated using Google Generative AI.
These embeddings are indexed in a FAISS vector database.

**Ask Questions:**
Enter your query in the text input box.
The system retrieves relevant content and generates an answer using Google Generative AI.

**View Results:**
The app displays the generated answer, contextually based on the content of the uploaded PDF.

### 🤝 Contributions
Contributions are welcome! Feel free to open issues or submit pull requests to enhance the functionality or address any bugs.