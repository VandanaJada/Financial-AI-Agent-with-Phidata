from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import phi.api
from phi.model.openai import OpenAIChat


import os
import phi
from phi.playground import Playground, serve_playground_app
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

phi.api = os.getenv("PHI_API_KEY")


## Web Search Agent
websearch_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model =Groq(id = "llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions =["Always include sources"],
    show_tools_calls=True,
    markdown=True
)

##Financial Agent
finance_agent = Agent(
    name="Finance Agent",
    role="Finance AI Agent",
    model=Groq(id = "llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    show_tool_calls=True,
    description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    instructions=["Format your response using markdown and use tables to display data where possible."],
)

app = Playground(agents=[finance_agent,websearch_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app",reload=True)
