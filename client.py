import requests

url = "http://127.0.0.1:8000/speech"
data = {
    "prompt": "Hey, how are you doing today?",
    "description": "Jon's voice is monotone yet normal in delivery, with a very close recording that almost has no background noise.",
}

response = requests.post(url, json=data)

# Save the output to a file
with open("output.wav", "wb") as f:
    f.write(response.content)
