import streamlit as st
from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import tool

from todoist_api_python.api import TodoistAPI

# ------------------ ENV SETUP ------------------
load_dotenv()

todoist_api_key = os.getenv("TODOIST_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

todoist = TodoistAPI(todoist_api_key)

# ------------------ TOOLS (WITH DOCSTRINGS) ------------------
@tool
def add_task(task: str, desc: str = None):
    """Add a new task to the user's Todoist list."""
    todoist.add_task(content=task, description=desc)


@tool
def show_tasks():
    """Show all tasks from the user's Todoist list."""
    results = todoist.get_tasks()
    tasks = []
    for task_list in results:
        for task in task_list:
            tasks.append(task.content)
    return tasks


tools = [add_task, show_tasks]

# ------------------ LLM ------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.0-pro",
    google_api_key=gemini_api_key,
    temperature=0.3,
    max_retries=1
)

# ------------------ SYSTEM PROMPT ------------------
system_prompt = """
You are a general-purpose AI assistant AND a task manager.

GENERAL RULES:
- You are fully allowed to answer general knowledge questions.
- If the question is NOT about tasks, answer it normally.
- NEVER refuse general knowledge questions.

TASK MANAGEMENT RULES:
- Use add_task ONLY when the user clearly wants to create or add a task.
- Use show_tasks ONLY when the user asks to see or list tasks.

TASK DISPLAY RULE:
- When showing tasks, ALWAYS display them in BULLET POINTS.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("history"),
        ("user", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False
)

# ------------------ STREAMLIT UI ------------------
st.set_page_config(
    page_title="AI Task Manager",
    page_icon="📝",
    layout="wide"
)

st.title("📝 AI Task Manager Assistant")

# ------------------ SESSION STATE ------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------ SIDEBAR: TODO LIST ------------------
st.sidebar.title("📋 Your Todo List")

try:
    todo_items = show_tasks.invoke({})  # ✅ FIXED
    if todo_items:
        for task in todo_items:
            st.sidebar.markdown(f"• {task}")
    else:
        st.sidebar.info("No tasks available.")
except Exception as e:
    st.sidebar.error("Failed to load tasks.")

# ------------------ CHAT HISTORY ------------------
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**You:** {msg.content}")
    else:
        st.markdown(f"**Assistant:** {msg.content}")

st.divider()

# ------------------ USER INPUT ------------------
user_input = st.text_input("Ask anything or manage tasks:")

if st.button("Send") and user_input:
    response = agent_executor.invoke(
        {
            "input": user_input,
            "history": st.session_state.history
        }
    )

    st.session_state.history.append(HumanMessage(content=user_input))
    st.session_state.history.append(AIMessage(content=response["output"]))

    st.rerun()
