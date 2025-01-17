# PDF to FAISS Query System using Google AI Studio
This project showcases a PDF-based Retrieval-Augmented Generation (RAG) system. It extracts text from uploaded PDF documents, embeds the text using Google AI Studio's embedding models, indexes the embeddings in a FAISS vector database, and answers user queries based on the indexed content. The app is implemented with Streamlit for an interactive user interface.

### 🚀 Features
- Extracts text from PDF documents using PyPDF2.
- Chunks extracted text into manageable pieces for embedding.
- Uses Google AI Studio's embedding model (text-embedding-004) to generate vector embeddings.
- Stores embeddings in a FAISS vector database for efficient similarity search.
- Retrieves relevant content and generates answers using Google AI's Gemini-2.0-flash-exp generative model.
- Provides a user-friendly interface via Streamlit.

### 💡 Usage
**Upload a PDF:**
Use the file uploader to upload a PDF document.

**Process the PDF:**
The app extracts and chunks the text, generates embeddings, and indexes them in FAISS.

**Ask a Question:**
Enter your question in the text input box.
The system retrieves the most relevant chunks and generates an answer using Google Generative AI.

**View the Answer:**
The app displays the generated answer based on the uploaded PDF.

### 🤝 Contributions
Contributions are welcome! Feel free to open issues or submit pull requests to enhance the functionality or address any bugs.