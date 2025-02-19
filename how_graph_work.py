import logging
from typing import TypedDict, Literal
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Define the state structure
class ContentState(TypedDict):
    input_prompt: str
    content: str
    comment_tone: str = "student"
    grammar_checked: bool
    tone_adjusted: bool = False
    review_passed: bool

# Initialize the ChatOllama model
ollama_model = ChatOllama(model="mistral-nemo")

# Function: Generate Short Content
def generate_content(state: ContentState) -> ContentState:
    if state["tone_adjusted"]:
        response = ollama_model.invoke(f"Generate a short content piece: {state['input_prompt']} here is comment {state['comment_tone']}")
    else:
        response = ollama_model.invoke(f"Generate a short content piece: {state['input_prompt']}")
    state["content"] = response.content
    logging.info(f"Generated Content: {state['content']}")
    return state

# Function: Check Grammar
def check_grammar(state: ContentState) -> ContentState:
    response = ollama_model.invoke(f"Check grammar correctness: {state['content']}")
    state["content"] = response.content
    state["grammar_checked"] = True
    logging.info(f"comment_grammar: {state['content']}")
    return state

# Function: Adjust Tone (Optional)
def adjust_tone(state: ContentState) -> ContentState:
    response = ollama_model.invoke(f"Adjust tone for a {state['input_prompt']} context: {state['content']}")
    state["comment_tone"] = response.content
    state["tone_adjusted"] = True
    logging.info(f"comment_tone: {state['comment_tone']}")
    return state

# Function: Review Content
def review_content(state: ContentState) -> ContentState:
    response = ollama_model.invoke(f"Review this content for clarity and effectiveness: {state['content']}")
    state["review_passed"] = "Approved" in response.content  # Assume model responds with "Approved" or "Needs revision"
    logging.info(f"Generated Content: {state['content']}")
    return state

# Function: Decide Next Step (Conditional Edge)
def decide_next_step(state: ContentState) -> str:
    if state["review_passed"]:
        return "Final"
    return "Generate"

# Create the state graph
graph = StateGraph(ContentState)

# Add nodes to the graph
graph.add_node("Generate", generate_content)
graph.add_node("GrammarCheck", check_grammar)
graph.add_node("ToneAdjust", adjust_tone)
graph.add_node("Review", review_content)
graph.add_node("Final", lambda state: state)

# Define standard edges (parallel execution)
graph.add_edge(START, "Generate")
graph.add_edge("Generate", "GrammarCheck")
graph.add_edge("Generate", "ToneAdjust")

# Define edges leading to review after grammar and tone adjustments
graph.add_edge("GrammarCheck", "Review")
graph.add_edge("ToneAdjust", "Review")

# Conditional edge: If review fails, regenerate content; if it passes, finalize.
graph.add_conditional_edges("Review", decide_next_step)

# Initialize the state
app = graph.compile()
initial_state = ContentState(
    input_prompt="Write a short introduction for a tech blog.",
    content="",
    grammar_checked=False,
    tone_adjusted=False,
    review_passed=False
)

# Execute the graph
final_state = app.invoke(initial_state)

# Output the final state
print(final_state['response'])
