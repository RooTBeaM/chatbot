# Initialize LLM

# from langchain_openai import ChatOpenAI
# LLM = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)

# from langchain_google_genai import ChatGoogleGenerativeAI
# LLM = ChatGoogleGenerativeAI( model="gemini-1.5-flash-exp-0827", temperature=0)

from langchain_ollama import ChatOllama
# LLM = ChatOllama(model="llama3.2",temperature=0.15)
LLM = ChatOllama(model="mistral",temperature=0.2)