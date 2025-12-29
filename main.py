
from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor

from todoist_api_python.api import TodoistAPI


load_dotenv()

todoist_api_key = os.getenv("TODOIST_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

todoist = TodoistAPI(todoist_api_key)


@tool
def add_task(task, desc=None):
    """
    Add a new task to the user's Todoist list.
    Use when the user clearly wants to create a task.
    """
    todoist.add_task(
        content=task,
        description=desc
    )


@tool
def show_tasks():
    """
    Show all tasks from Todoist.
    Use this tool when user wants to see their tasks.
    """
    results_paginator = todoist.get_tasks()
    tasks = []

    for task_list in results_paginator:
        for task in task_list:
            tasks.append(task.content)

    return tasks


tools = [add_task, show_tasks]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # ✅ supported + lowest quota usage
    google_api_key=gemini_api_key,
    temperature=0.2,
    max_retries=0             # ✅ stops retry spam
)



system_prompt = """
You are a general-purpose AI assistant AND a task manager.

GENERAL RULES:
- You are fully allowed to answer general knowledge questions
  (history, science, explanations, definitions, facts, etc.).
- If the question is NOT about tasks, answer it normally in plain text.
- NEVER refuse general knowledge questions.

TASK MANAGEMENT RULES:
- Use add_task ONLY when the user clearly wants to create or add a task.
- Use show_tasks ONLY when the user asks to see, list, or show tasks.

TASK DISPLAY RULE:
- When showing tasks, ALWAYS display them in clear BULLET POINTS.
- Each task must appear on a new line starting with a bullet (• or *).
- Do NOT add explanations before or after the task list unless asked.

IMPORTANT:
- Do not say "my purpose is only task management".
- Be helpful, clear, and concise.
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


history = []

while True:
    user_input = input("you: ")

    response = agent_executor.invoke(
        {
            "input": user_input,
            "history": history
        }
    )

    print(response["output"])

    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response["output"]))
