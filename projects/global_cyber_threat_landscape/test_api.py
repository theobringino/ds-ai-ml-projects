import requests
import json



# The URL for your Flask endpoint
API_URL = "http://127.0.0.1:5000/predict" 

# Example data point to send to the API. 
# NOTE: This must contain all the original feature columns the model expects.
test_data = {
    "Country": "United States of America",
    "Spam": 0.0012,
    "Ransomware": 0.0003,
    "Exploit": 0.0005,
    "Malicious Mail": 0.0025,
    "Network Attack": 0.015,
    "Web Threat": 0.016
}

# The request headers specify that you are sending JSON data
headers = {
    "Content-Type": "application/json",
    "X-API-Key" : "23Theo23APIKey"
}

try:
    # Send the POST request
    response = requests.post(API_URL, data=json.dumps(test_data), headers=headers)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        print("API Call Successful!")
        # Print the JSON response from the API
        print(json.dumps(response.json(), indent=4))
        
        # Example check:
        prediction = response.json().get('local_infection_rate_prediction')
        print(f"\nPredicted Local Infection Rate: {prediction:.6f}")
        
    else:
        print(f"API Call Failed. Status Code: {response.status_code}")
        print(f"Response: {response.text}")

except requests.exceptions.ConnectionError:
    print(f"Connection Error: Ensure your 'api.py' is running on {API_URL}.")