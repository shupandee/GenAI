import requests
import json
import warnings

# Suppress warnings from the requests library for cleaner output
warnings.filterwarnings("ignore")

# --- Configuration ---
# The URL of your running FastAPI service
# Adjust the port if your server is running on a different one
API_URL = "https://3bbfaabc966c.ngrok-free.app/api/v1/query"

def run_policy_analysis():
    """
    Sends a POST request with a specific PDF URL and a list of questions
    to the FastAPI server and prints the structured response.
    """
    # The URL of the policy document to be analyzed at runtime.
    pdf_url = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"
    
    # A list of questions related to the policy.
    questions = [
        "What is the definition of 'Accident' in this policy?",
    ]

    # The payload for the POST request, matching the DynamicQueryRequest Pydantic model.
    payload = {
        "documents": pdf_url,
        "questions": questions
    }
    
    # The headers for the request.
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    try:
        print("--- Sending POST request to the API ---")
        # Send the POST request. A timeout is set for robust error handling.
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=300)
        
        # Check for a successful response (status code 200).
        response.raise_for_status()

        # Parse the JSON response.
        response_data = response.json()
        
        # Print the structured output for each question.
        print("\n--- API Response ---")
        print(json.dumps(response_data, indent=2))
        
    except requests.exceptions.RequestException as e:
        print(f"\nError: An error occurred while making the request.")
        print(f"Details: {e}")
        print("\nPlease ensure your FastAPI server is running and accessible at the specified URL.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    run_policy_analysis()