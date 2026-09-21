from phi.agent import Agent
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from phi.model.groq import Groq
from phi.playground import Playground, serve_playground_app
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

app=Playground(
    agents=[web_seaarch_agent, financial_agent]
).get_app()

if __name__ == "__main__":
    serve_playground_app(app)