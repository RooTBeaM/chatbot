import requests
import json

# ollama api
url = 'http://localhost:11434/api/generate'

data = {
    # "model": "llama3.2",
    "model": "Dream",
    "prompt": "Who is Albert Einstein? what is the theory of relativity?"
}

response = requests.post(url, json=data, stream=True)
# check the response status 
if response.status_code == 200:
    print("Generated text:\n", end="", flush=True)
    for line in response.iter_lines():
        if line:
            # Decode the line and parse the json
            line = line.decode('utf-8')
            result = json.loads(line)
            # Generate the text from the response
            generated_text = result.get("response", "")
            print(generated_text, end="", flush=True)
else:
    print("Error:", response.status_code)
    print(response.text)
