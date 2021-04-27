import requests 

apiURL = "http://127.0.0.1:5000/recommend_api"

postRequest = requests.post(apiURL, json = {'movie': 'Toy Story'})

print(postRequest.json())