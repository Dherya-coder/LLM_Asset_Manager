import requests
import json

def test_allocation_api():
    url = "http://localhost:8000/allocate-investment"
    
    # Test case 1: Young investor with moderate amount
    payload1 = {
        "age": 25,
        "investment_amount": 50000
    }
    
    # Test case 2: Middle-aged investor with large amount
    payload2 = {
        "age": 45,
        "investment_amount": 500000
    }
    
    # Test case 3: Senior investor with conservative amount
    payload3 = {
        "age": 65,
        "investment_amount": 100000
    }
    
    # Test all cases
    test_cases = [
        ("Young Investor", payload1),
        ("Middle-aged Investor", payload2),
        ("Senior Investor", payload3)
    ]
    
    for case_name, payload in test_cases:
        print(f"\nTesting {case_name}:")
        print(f"Input: {json.dumps(payload, indent=2)}")
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            result = response.json()
            print("\nResponse:")
            print(f"Allocation: {json.dumps(result['allocation_percentage'], indent=2)}")
            print(f"Recommendation: {result['recommendation']}")
            
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_allocation_api() 