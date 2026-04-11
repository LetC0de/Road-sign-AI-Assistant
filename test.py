import requests

url = "http://127.0.0.1:5000/predict"

files = {
    "image": open("2.jpg", "rb")  # put any test image
}

response = requests.post(url, files=files)

print(response.json())