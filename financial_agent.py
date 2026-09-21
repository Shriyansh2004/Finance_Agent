from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools 
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
load_dotenv()  # Load environment variables from .env file

# This is my web search agent
web_seaarch_agent=Agent(
    name="Web Search Agent",
    role="Searches the web for information",
    model=Groq(id="openai/gpt-oss-120b"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True,

)

# This is my financial agent
financial_agent=Agent(
    name="Financial Agent",
    role="Analyzes financial data and provides insights",
    model=Groq(id="openai/gpt-oss-120b"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True),
        ],
    instructions=["Use tables to display the data"],
    show_tools_calls=True,
    markdown=True,
)

multi_ai_agent=Agent(
    team=[web_seaarch_agent, financial_agent],
    name="Multi AI Agent",
    model=Groq(id="openai/gpt-oss-120b"),
    instructions=["Always include sources", "Use tables to display the data"],
    show_tools_calls=True,
    markdown=True
)

financial_agent.print_response(
    "Summarize analyst recommendations and share the latest news for NVDA. Keep the response concise.",
    stream=True,
)