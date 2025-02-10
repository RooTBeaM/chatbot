import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import LLM
from chain.prompts import PLANNING_PROMPT

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Create a PromptTemplate
plan_prompt = ChatPromptTemplate([
        ('user', PLANNING_PROMPT)
    ])

plan_chain = plan_prompt | LLM | StrOutputParser()

## For testing
if __name__ == "__main__":
    # Test the plan_chain
    test_instruction = "Write a current and up to date 100% unique guide for my intermittent fasting for women over 50 cookbook on \u201cSnacks\u201d with humanlike style, using transitional phrases, and avoidance of unnatural sentence structure while explaining in details extensively and comprehensively."
    
    # Invoke the plan_chain
    result = plan_chain.invoke({"instructions": test_instruction})
    
    # Print the result
    print(f"LLM model : {plan_chain.__dict__['middle']}")
    print("Generated Writing Plan:")
    print(result)

