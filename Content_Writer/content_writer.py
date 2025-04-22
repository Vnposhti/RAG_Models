from typing import List, TypedDict, Annotated, Literal
from dataclasses import dataclass
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
import operator
from datetime import datetime
from pathlib import Path
from langgraph.graph import START, StateGraph, END
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

# Configure page
st.set_page_config(page_title="Academic Content Generator", layout="wide")

# Initialize LLM
@st.cache_resource
def init_llm():
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash")

llm = init_llm()

# Base schemas for content structure
@dataclass
class Subtopic:
    """Schema for a single subtopic"""
    name: str
    description: str
    difficulty_level: Literal["basic", "intermediate", "advanced"] = "basic"

class ContentSection(BaseModel):
    """Schema for a content section"""
    title: str = Field(..., description="Title of the content section")
    content: str = Field(..., description="Main content text")
    theory: str = Field(..., description="Theoretical explanation")
    practical_examples: List[str] = Field(default_factory=list, description="Practical examples and implementations")
    key_points: List[str] = Field(default_factory=list, description="Important concepts to remember")
    references: List[str] = Field(default_factory=list, description="Academic references and citations")

class ContentPlan(BaseModel):
    """Schema for academic content generation plan"""
    topic: str = Field(..., description="Main topic to generate content for")
    subtopics: List[Subtopic] = Field(..., description="List of subtopics to cover")
    content_style: Literal["academic"] = Field(
        default="academic",
        description="Academic style content with detailed explanations"
    )
    learning_objectives: List[str] = Field(
        default_factory=list,
        description="Specific learning objectives for the topic"
    )

# State management schemas
class State(TypedDict):
    """Overall workflow state"""
    topic: str
    subtopics: List[Subtopic]
    study_material: str
    completed_sections: Annotated[List[str], operator.add]

class WorkerState(TypedDict):
    """Individual worker state"""
    subtopic: Subtopic
    completed_sections: Annotated[List[str], operator.add]

# Content generation schemas
class ContentFormat(BaseModel):
    """Schema for academic content formatting specifications"""
    use_markdown: bool = Field(default=True, description="Whether to use markdown formatting")
    include_headers: bool = Field(default=True, description="Whether to include section headers")
    code_style: str = Field(default="github", description="Style for code blocks")
    include_equations: bool = Field(default=True, description="Whether to include mathematical equations")
    include_diagrams: bool = Field(default=True, description="Whether to include diagrams and illustrations")

class ContentRequest(BaseModel):
    """Schema for academic content generation request"""
    subtopic: Subtopic
    format_specs: ContentFormat = Field(default_factory=ContentFormat)
    max_length: int = Field(default=10000, description="Maximum content length in characters")
    include_examples: bool = Field(default=True, description="Whether to include examples")
    include_practice_problems: bool = Field(default=True, description="Whether to include practice problems")

class ContentResponse(BaseModel):
    """Schema for generated academic content"""
    section_title: str = Field(..., description="Title of the generated section")
    learning_objectives: List[str] = Field(..., description="Learning objectives for this section")
    prerequisites: List[str] = Field(default_factory=list, description="Required prerequisite knowledge")
    main_content: str = Field(..., description="Main content text")
    theoretical_background: str = Field(..., description="Detailed theoretical explanation")
    examples: List[str] = Field(default_factory=list, description="Detailed examples with explanations")
    practice_problems: List[str] = Field(default_factory=list, description="Practice problems with solutions")
    key_takeaways: List[str] = Field(default_factory=list, description="Key points to remember")
    metadata: dict = Field(default_factory=dict, description="Additional content metadata")

# Node functions
def orchestrator(state: State):
    """Orchestrator that generates detailed subtopics for the given topic."""
    
    # Generate subtopics using LLM
    response = llm.invoke(
        [
            SystemMessage(content="You are a curriculum designer. Generate all possible detailed subtopics for the given topic. "
                                "For each subtopic, provide a name and brief description."),
            HumanMessage(content=f"Generate subtopics for: {state['topic']}")
        ]
    )
    
    # Parse response into Subtopic objects
    # Assuming response is in format: "Name: xxx\nDescription: yyy\n\n"
    subtopics_text = response.content.split("\n\n")
    subtopics = []
    
    for st in subtopics_text:
        if not st.strip():
            continue
        lines = st.split("\n")
        if len(lines) >= 2:
            name = lines[0].replace("Name:", "").strip()
            desc = lines[1].replace("Description:", "").strip()
            subtopics.append(Subtopic(name=name, description=desc))
    
    return {"subtopics": subtopics}

def content_writer(state: WorkerState) -> dict:
    """Worker that writes academic content for each subtopic"""
    
    # Create content request
    request = ContentRequest(
        subtopic=state['subtopic'],
        format_specs=ContentFormat(
            use_markdown=True,
            include_headers=True,
            code_style="github",
            include_equations=True,
            include_diagrams=True
        ),
        include_examples=True,
        include_practice_problems=True
    )
    
    response = llm.invoke(
        [
            SystemMessage(content=(
                "Create comprehensive academic content with detailed explanations. "
                "Include theoretical background, practical examples, and practice problems. "
                "Use clear language and provide step-by-step explanations. "
                "Include mathematical notations and diagrams where appropriate."
            )),
            HumanMessage(content=(
                f"Generate academic content for: {request.subtopic.name}\n\n"
                f"Description: {request.subtopic.description}\n"
                f"Difficulty Level: {request.subtopic.difficulty_level}"
            ))
        ]
    )
    
    return {
        "content": response.content,
        "examples": [],  # Extract examples if needed
        "practice_problems": [],  # Extract practice problems if needed
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "difficulty_level": request.subtopic.difficulty_level
        }
    }

def synthesizer(state: State):
    """Combines all sections into final study material."""
    
    # Join all completed sections with newlines
    combined_content = "\n\n".join(state["completed_sections"])
    
    # Add title and introduction
    final_content = f"# Study Material: {state['topic']}\n\n{combined_content}"
    
    return {"study_material": final_content}

# Create workflow
def create_workflow():
    """Creates the content generation workflow"""
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator)
    workflow.add_node("content_writer", content_writer)
    workflow.add_node("synthesizer", synthesizer)
    
    # Add edges
    workflow.add_edge(START, "orchestrator")
    
    # Add conditional edges for parallel content writing
    def assign_workers(state: State) -> List[dict]:
        return [
            {
                "worker": "content_writer",
                "state": {
                    "subtopic": st,
                    "completed_sections": []
                }
            }
            for st in state["subtopics"]
        ]
    
    workflow.add_conditional_edges(
        "orchestrator",
        assign_workers,
        ["content_writer"]
    )
    
    workflow.add_edge("content_writer", "synthesizer")
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()

def main():
    st.title("Content Developer")
   # Main content area
    topic = st.text_input("Enter your main topic:")
    
    # Sidebar content
    with st.sidebar:
        workflow_image_path = Path("output.png")
        st.image(str(workflow_image_path), 
                caption="Content Generation Workflow",
                width=240)
    
    if st.button("Generate Content"):
        if not topic:
            st.error("Please enter a topic first!")
            return
            
        with st.spinner("Generating subtopics..."):
            # Create initial state
            state = {"topic": topic, "subtopics": [], "study_material": "", "completed_sections": []}
            
            # Generate subtopics
            subtopics_result = orchestrator(state)
            subtopics = subtopics_result["subtopics"]
                 
            # Generate content for each subtopic
            st.subheader(topic)
            
            for subtopic in subtopics:
                with st.spinner(f"Generating content for {subtopic.name}..."):
                    worker_state = {
                        "subtopic": subtopic,
                        "completed_sections": []
                    }
                    
                    # Generate content
                    content = content_writer(worker_state)
                    
                    # Display content in expandable section
                    with st.expander(f"{subtopic.name}"):
                        st.markdown(content["content"])
                        
                        if "examples" in content and content["examples"]:
                            st.subheader("Examples")
                            for example in content["examples"]:
                                st.write(example)
                                
                        if "practice_problems" in content and content["practice_problems"]:
                            st.subheader("Practice Problems")
                            for problem in content["practice_problems"]:
                                st.write(problem)

if __name__ == "__main__":
    main()




