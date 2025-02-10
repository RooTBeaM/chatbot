import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain.plan import plan_chain

def planning_node(state):
    """take the initial prompt and write a plan to make a long doc"""
    print("---PLANNING THE WRITING---")
    print(f"LLM model : {plan_chain.__dict__['middle']}")
    initial_prompt = state['initial_prompt']
    num_steps = int(state['num_steps'])
    num_steps += 1

    plan = plan_chain.invoke({"instructions": initial_prompt})
    # print(plan)

    return {"plan": plan, "num_steps":num_steps}