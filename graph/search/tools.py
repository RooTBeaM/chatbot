from langchain_community.tools import DuckDuckGoSearchRun
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

search = DuckDuckGoSearchRun(backend="news")
tools = [add, multiply, divide, search, get_stock_price]
# tools = [add, multiply, divide, search]
