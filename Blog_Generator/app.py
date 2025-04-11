import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(model="llama3-70b-8192")

# Function to generate 10 blog titles
def generate_titles(topic):
    response = llm.invoke(f"Generate 10 engaging blog titles on: {topic}")
    titles = [title.strip() for title in response.content.split("\n") if title.strip()]
    return titles[:10]  # Ensure only 10 titles are returned

# Function to generate blog content based on selected title
def generate_content(selected_title):
    response = llm.invoke(f"Write a detailed, well-structured blog on: {selected_title}")
    return response.content

# Streamlit UI
st.title("📝 AI Blog Generator")

# User Input for Topic
topic = st.text_input("Enter a topic: ")

# Generate Titles Button
if st.button("Generate Titles"):
    st.session_state["titles"] = generate_titles(topic)

# Display Titles for Selection
if "titles" in st.session_state:
    st.subheader("📌 Select a Title:")
    selected_title = st.radio("Generated Titles:", st.session_state["titles"])

    # Option to Regenerate Titles
    if st.button("Regenerate Titles"):
        st.session_state["titles"] = generate_titles(topic)
        st.session_state.pop("blog_content", None)  # Clear previous content

    # Generate Blog Button
    if st.button("Generate Blog"):
        st.session_state["blog_content"] = generate_content(selected_title)
        st.session_state["selected_title"] = selected_title

# Show Generated Blog
if "blog_content" in st.session_state:
    st.subheader("📖 Generated Blog")
    st.markdown(f"### {st.session_state['selected_title']}")
    st.markdown(st.session_state["blog_content"])