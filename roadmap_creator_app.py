import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is None:
    st.error("GROQ_API_KEY environment variable is not set.")
    st.stop()
os.environ["GROQ_API_KEY"] = groq_api_key

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Define State
class State(TypedDict):
    name:str
    career:str
    repos:str
    tech_stack:str
    summary:str

def top_github_repos(state:State):
    msg=llm.invoke(f"""
                   You are an expert roadmap creator.
                   Generate a precise roadmap for user whose name is {state["name"]}.
                   User wants to build a career in {state['career']}.
                   Fetch the link of top GitHub repositories which provides roadmap based on {state['career']} in Markdown formatting.
    """)
    return {"repos":msg.content}

def list_tech_stack(state:State):
    msg=llm.invoke(f"""
                   Based on provided {state['career']}, create a list of technical skills required to master it.
                   Create a bullet-wise list with Markdown formatting. 
    """)
    return {"tech_stack":msg.content}

def generate_summary(state:State):
    msg=llm.invoke(f"""
                   After fetching top GitHub Repos from {state['repos']} and technical skill requirements from {state['tech_stack']},
                   Now generate a brief summary to guide {state['name']} based on the given Roadmap.
                   Use Markdown formatting.
    """)
    return {"summary":msg.content}

# Build the graph
graph = StateGraph(State)
graph.add_node("GitHub Repos", top_github_repos)
graph.add_node("Tech Stack", list_tech_stack)
graph.add_node("Summarize", generate_summary)
graph.add_edge(START,"GitHub Repos")
graph.add_edge("GitHub Repos","Tech Stack")
graph.add_edge("Tech Stack","Summarize")
graph.add_edge("Summarize",END)
compiled_graph = graph.compile()

# Streamlit UI
st.set_page_config(page_title="Roadmap Creator", page_icon=":briefcase:", layout="wide")
st.title("Roadmap Creator Agent")
st.write("Enter your details to get a personalized roadmap:")

with st.form("career_form"):
    name = st.text_input("Name")
    career = st.text_input("Career Goal")
    submitted = st.form_submit_button("Get Roadmap")

if submitted:
    if not all([name, career]):
        st.warning("Please fill in all fields.")
    else:
        with st.spinner("Generating your roadmap..."):
            state = compiled_graph.invoke({
                "name": name,
                "career": career
            })
        st.markdown(state["repos"])
        st.divider()
        st.markdown(state["tech_stack"])
        st.divider()
        # Add download button for the career plan
        st.download_button(
            label="Download your Roadmap as Markdown",
            data=state["summary"],
            file_name=f"roadmap_{name.replace(' ', '_')}.md",
            mime="text/markdown"
        ) 