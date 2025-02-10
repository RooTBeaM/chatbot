from graph import create_workflow
from llm import LLM

# Create the workflow
app = create_workflow()

# test instruction
test_instruction = "You are a financial and mathematical assistant specializing in solving complex numerical problems, performing precise calculations, and interpreting financial data. Your role is to provide accurate results, explain steps clearly, and ensure all responses are logically sound and contextually relevant to the given task."
# Run the workflow
inputs = {"initial_prompt": test_instruction,  
          "num_steps": 0,
        #   "query" : "What is the stock price of the company that Jensen Huang is CEO of?",
          "query" : "how much cost if I buy a apple watch including apple care +?",
          "llm_name": LLM.model}

print(f"LLM model : {inputs['llm_name']}")
print(f"Task : {inputs['initial_prompt']}")

output = app.invoke(inputs)

# print(output['final_answer'])
for m in output['messages']:
    m.pretty_print()