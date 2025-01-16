import time
import asyncio
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition  # Checker for tool responses
from langgraph.prebuilt import ToolNode
import yfinance as yf

def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    This tool takes two integers as input and returns their product. 
    Useful for calculating the result of multiplication operations.

    Args:
        a: The first integer to multiply.
        b: The second integer to multiply.
    """
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers.

    This tool takes two integers as input and returns their sum. 
    Useful for combining values or performing addition operations.

    Args:
        a: The first integer to add.
        b: The second integer to add.
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide one integer by another.

    This tool divides the first integer by the second and returns the quotient as a float. 
    Use this for division operations where the result may be fractional.

    Args:
        a: The numerator.
        b: The denominator (must not be zero).
    """
    return a / b


def get_stock_price(ticker: str) -> float:
    """Gets a stock price from Yahoo Finance.

    Args:
        ticker: ticker str
    """
    # """This is a tool for getting the price of a stock when passed a ticker symbol"""
    stock = yf.Ticker(ticker)
    return stock.info['previousClose']

search = DuckDuckGoSearchRun()
tools = [add, multiply, divide, search, get_stock_price]

# LLM Model
llm = ChatOllama(model="llama3.2", temperature=0.3)
sys_msg = SystemMessage(content="You are a helpful assistant tasked with using search and performing arithmetic on a set of inputs.")
llm_with_tools = llm.bind_tools(tools)

# Node Function
def reasoner(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build the Graph
builder = StateGraph(MessagesState)
builder.add_node("reasoner", reasoner)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "reasoner")
builder.add_conditional_edges("reasoner", tools_condition)
builder.add_edge("tools", "reasoner")
react_graph = builder.compile()

# Function to send a single request and measure latency
async def send_request(message):
    start_time = time.perf_counter()
    messages = react_graph.invoke({"messages": [message]})
    latency = time.perf_counter() - start_time
    return latency, messages

# Test Function for Concurrent Requests
async def test_latency(num_requests):
    tasks = [
        send_request(HumanMessage(content=f"Request {i}: What is the price of MacBook now? If discounted by {i*5}%, what would it be?"))
        for i in range(1, num_requests + 1)
    ]
    results = await asyncio.gather(*tasks)
    for i, (latency, response) in enumerate(results, start=1):
        print(f"Request {i} latency: {latency:.2f}s, Response: \n{response['messages'][-1].content}")

# Entry point
if __name__ == "__main__":
    num_requests = 10  # Adjust number of concurrent requests for testing
    asyncio.run(test_latency(num_requests))
