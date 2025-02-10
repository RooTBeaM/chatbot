import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain.writing import write_chain
import re

def count_words(text):
        """
        Count the number of words in the given text.
        
        Args:
            text (str): The input text to count words from.
        
        Returns:
            int: The number of words in the text.
        """
        # Split the text into words and count them
        words = text.split()
        return len(words)

def writing_node(state):
    """take the initial prompt and write a plan to make a long doc"""
    print("---WRITING THE DOC---")
    print(f"LLM model : {write_chain.__dict__['middle']}")
    initial_instruction = state['initial_prompt']
    plan = state['plan']
    num_steps = int(state['num_steps'])
    num_steps += 1

    txt = re.sub(r"\*\*", "", plan)
    plan = txt.strip().replace('\n\n', '\n')
    txt_ls = plan.split('\n')
    planning_steps = [x for x in txt_ls if "paragraph" in x.lower()]

    text = ""
    responses = []
    if len(planning_steps) > 50:
        print("plan is too long")
        print(plan)
        return None
    for idx,step in enumerate(planning_steps):
        # result = step
        print(f"----------------------------{idx}----------------------------")
        print(step)
        print("----------------------------\n\n")
        # Invoke the write_chain
        result = write_chain.invoke({
            "instructions": initial_instruction,
            "plan": plan,
            "text": text,
            "STEP": step
        })
        responses.append(result)
        text += result + '\n\n'

    final_doc = '\n\n'.join(responses)

    # Count words in the final document
    word_count = count_words(final_doc)
    print(f"Total word count: {word_count}")

    return {"final_doc": final_doc, "word_count": word_count, "num_steps":num_steps}



