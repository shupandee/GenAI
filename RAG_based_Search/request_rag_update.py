import requests
import json

# --- Configuration ---
# By default, FastAPI runs on http://127.0.0.1:8000
API_URL = "http://127.0.0.1:8000/api/v1/query"

# --- Function to send a query ---
def send_rag_query(query_text: str):
    """
    Sends a POST request to the RAG API with the given query.
    Prints the response status and content.
    """
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "query": query_text
    }

    print(f"\n--- Sending Query: '{query_text}' ---")
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        print(f"Status Code: {response.status_code}")
        response_json = response.json()
        print("Response JSON:")
        print(json.dumps(response_json, indent=2))

    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
        print(f"Response Body: {errh.response.text}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
        print("Please ensure your FastAPI server is running at", API_URL)
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"An unexpected error occurred: {err}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON response. Raw response: {response.text}")

# --- Main execution block ---
if __name__ == "__main__":
    # --- Runtime Query Input ---
    print("\n--- Enter your custom queries interactively ---")
    while True:
        custom_query = input("Enter your claim query (or type 'exit' to quit): ")
        if custom_query.lower() == 'exit':
            break
        send_rag_query(custom_query)

    print("\nScript finished.")