import requests
import json

# Update the URL to include the correct endpoint
url = "http://127.0.0.1:8080/v1/complete"  # Change '/v1/complete' to the correct endpoint

# Define the data payload
data = {
    "prompt": "Hello, how are you?",
    "max_tokens": 50
}

# Define the headers
headers = {
    "Content-Type": "application/json"
}

# Send the POST request with the JSON data
response = requests.post(url, headers=headers, json=data)

# Print the status code and the response text
print("Status Code:", response.status_code)
print("Response:")
print(response.text)
