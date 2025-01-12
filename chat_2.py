# ollama with python
import ollama

response = ollama.list()

res = ollama.chat(
    model = "llama3.2", 
    messages = [{
        "role" : "user","content" : "Who is Albert Einstein? what is the theory of relativity?"
        }],
    stream = True
    )


for chunk in res:
    print(chunk['message']['content'], end="", flush=True)