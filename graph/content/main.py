import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from graph import create_workflow

# Load environment variables from .env file
load_dotenv()

# Create the workflow
app = create_workflow()

# test instruction
# test_instruction = """Write a 5000 words, current and up to date 100% unique guide for my intermittent fasting for women over 50 cookbook on \u201cSnacks\u201d with humanlike style, using transitional phrases, and avoidance of unnatural sentence structure while explaining in details extensively and comprehensively."""
test_instruction = """
Write a 5000 word, 
Objective:Create engaging and informative content related to AI and robotics. The content should be clear, insightful, and tailored to the target audience, whether they are beginners, enthusiasts, or experts.
Tone & Style:
 - Professional yet approachable
 - Concise and well-structured
 - Use simple explanations for complex topics when needed
 - Include real-world examples and applications
Example Topics:
 - How AI is Transforming Robotics
 - The Role of Machine Learning in Autonomous Systems
 - Ethical Concerns in AI-Powered Robotics
 - Future Trends in AI and Robotics
"""

# Run the workflow
inputs = {"initial_prompt": test_instruction,  
          "num_steps": 0,
          # "llm_name": "llama32"}
          "llm_name": "mistral"}
          # "llm_name": "deepseek-r1_8b"}

print(f"LLM model : {inputs['llm_name']}")
print(f"Task : {inputs['initial_prompt']}")
output = app.invoke(inputs)

# print(output['final_doc'])