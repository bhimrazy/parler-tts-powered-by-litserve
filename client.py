import requests

url = "http://127.0.0.1:8000/speech"
headers = {"Authorization": "Bearer lit", "Content-Type": "application/json"}
data = {
    "prompt": "Hey, how are you doing today?",
    "description": "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.",
}

response = requests.post(url, json=data)

# Save the output to a file
with open("output.wav", "wb") as f:
    f.write(response.content)
