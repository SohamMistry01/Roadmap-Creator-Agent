import os
import streamlit as st
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
github_token = os.getenv("GITHUB_TOKEN")

if not groq_key or not github_token:
    st.error("Missing GROQ_API_KEY or GITHUB_TOKEN in environment.")
    st.stop()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)

headers = {
"Authorization": f"token {github_token}",
"Accept": "application/vnd.github+json"
}

st.title("📍 Roadmap Generator")
st.write("Get a roadmap by exploring trending GitHub repositories in your chosen tech domain.")

domain = st.text_input("Enter your domain (e.g., Data Science, Web Development, DevOps)")

if st.button("Generate Roadmap"):
    if not domain:
        st.warning("Please enter a domain.")
    else:
        with st.spinner("Fetching trending GitHub repositories..."):
            query = f"{domain} in:name,description"
            url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc&per_page=5"
            res = requests.get(url, headers=headers)
            if res.status_code != 200:
                st.error("GitHub API error. Try again later.")
            else:
                items = res.json().get("items", [])
                repo_data = "\n".join(
                    [f"{i+1}. {repo['name']}: {repo['description'] or 'No description'}" for i, repo in enumerate(items)]
                )

                prompt = f""" 
                You are a career assistant AI.

                Using the following top GitHub repositories in the domain of {domain}:
                {repo_data}

                Please generate a step-by-step skill-based learning roadmap. Include:

                Key learning modules

                Topics to cover (basic to advanced)

                When and how to refer these repositories

                Certifications or projects to build
                Format the output using Markdown.
                """
                output = llm.invoke(prompt)
                st.success("Here's your personalized roadmap:")
                with st.expander("View Your Roadmap"):
                    st.markdown(output.content)