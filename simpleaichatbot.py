import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

load_dotenv()

llm = LLM(
    model='gemini/gemini-2.5-flash', 
    api_key=os.getenv("GEMINI_API_KEY")
)

qa_agent = Agent(
    role="Helping Chatbot",
    goal="Provide answer for question: {question}",
    backstory="You can chat and answer questions based on your knowledge.",
    tools=[],
    llm=llm
)

qa_task = Task(
    description="Answer the user question",
    expected_output="A get clear answer",
    agent=qa_agent
)

crew = Crew(
    agents=[qa_agent],
    tasks=[qa_task],
    verbose=True
)

result = crew.kickoff(inputs={'question': 'what is machine learning'})
print(result)