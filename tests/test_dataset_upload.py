import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_dataset_upload():
    print("Testing Dataset Upload...")
    
    # Create a dummy dataset
    dataset_payload = {
        "dataset": {
            "name": "Test Multi-Conv Dataset",
            "task": "Analyze customer support interactions",
            "conversations": [
                {
                    "initial_task": "Help user with login",
                    "turns": [
                        {
                            "user_message": "I cannot login",
                            "agent_interactions": [
                                {
                                    "agent_id": "support_agent",
                                    "agent_response": "Have you tried resetting your password?",
                                    "agent_steps": [
                                        {"thought": "User has login issue, suggesting password reset."}
                                    ]
                                }
                            ]
                        },
                        {
                            "user_message": "Yes, I did.",
                            "agent_interactions": [
                                {
                                    "agent_id": "support_agent",
                                    "agent_response": "Okay, please check your spam folder.",
                                    "agent_steps": [
                                        {"thought": "User tried reset, checking spam folder."}
                                    ]
                                }
                            ]
                        }
                    ]
                },
                {
                    "initial_task": "Help user with billing",
                    "turns": [
                        {
                            "user_message": "My bill is wrong",
                            "agent_interactions": [
                                {
                                    "agent_id": "billing_agent",
                                    "agent_response": "Can you provide your invoice number?",
                                    "agent_steps": [
                                        {"thought": "Billing issue, need invoice number."}
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        },
        "analysis_types": ["conversation", "trajectory"]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/agent/jobs", json=dataset_payload)
        result = {
            "status_code": response.status_code,
            "response": response.json()
        }
        with open("test_result.json", "w") as f:
            json.dump(result, f, indent=2)
            
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            job_id = response.json().get("job_id")
            print(f"Job ID: {job_id}")
            return job_id
        else:
            print("Failed to start job")
            return None
            
    except Exception as e:
        with open("test_result.json", "w") as f:
            json.dump({"error": str(e)}, f)
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    test_dataset_upload()
