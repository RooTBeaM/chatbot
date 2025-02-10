from langgraph.graph import StateGraph, END
from typing import Annotated,TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langgraph.prebuilt import tools_condition, ToolNode
import operator

from llm import LLM
from tools import tools
llm_with_tools = LLM.bind_tools(tools)

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        initial_prompt: initial prompt
        llm_name: name of the LLM
        num_steps: number of steps
        messages: state messages
        final_answer: answer
    """
    initial_prompt : str
    llm_name : str
    num_steps : int
    query : str
    messages : Annotated[list[AnyMessage], operator.add]
    final_answer : str

def reasoner(state):
    num_steps = int(state['num_steps'])
    num_steps += 1
    # System message
    sys_msg = SystemMessage(content=state['initial_prompt'])
    # Human message
    human_msg = HumanMessage(content=state["query"])
    # State message
    messages = state["messages"]
    result = [llm_with_tools.invoke([sys_msg,human_msg] + messages)]
    return {"messages":result, "final_answer":result, "num_steps":num_steps}

def create_workflow():
    workflow = StateGraph(GraphState)
    # Add nodes
    workflow.add_node("reasoner", reasoner)
    workflow.add_node("tools", ToolNode(tools))
    # Set entry point
    workflow.set_entry_point("reasoner")
    # Add edges
    workflow.add_conditional_edges("reasoner",tools_condition)
    workflow.add_edge("tools", "reasoner")
    return workflow.compile()