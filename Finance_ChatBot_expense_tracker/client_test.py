#!/usr/bin/env python3
"""
Simple Runner for Financial Assistant Demo
==========================================
Quick start script that doesn't require the full test suite.
"""

import requests
import json
import time
from datetime import datetime

class QuickDemo:
    """Simplified demo for quick testing"""
    
    def __init__(self):
        self.base_url = "http://127.0.0.1:8000"
        self.user_id = "quick_demo_user"
        self.device_id = None
    
    def check_api(self):
        """Check if API is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def chat(self, message):
        """Send chat message"""
        data = {
            "message": message,
            "user_id": self.user_id,
            "fetch_sms": False
        }
        
        try:
            response = requests.post(f"{self.base_url}/chat", json=data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response")
            else:
                return f"Error: HTTP {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def add_expense(self, amount, category, description):
        """Add expense directly"""
        data = {
            "amount": amount,
            "category": category,
            "description": description,
            "merchant": "Demo Store",
            "user_id": self.user_id,
            "source": "manual",
            "confidence_score": 1.0,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            response = requests.post(f"{self.base_url}/add-expense", json=data, timeout=10)
            if response.status_code == 200:
                return "Expense added successfully"
            else:
                return f"Failed to add expense: HTTP {response.status_code}"
        except Exception as e:
            return f"Error adding expense: {str(e)}"
    
    def get_expenses(self):
        """Get recent expenses"""
        try:
            response = requests.get(f"{self.base_url}/expenses/{self.user_id}?limit=5", timeout=10)
            if response.status_code == 200:
                expenses = response.json()
                if not expenses:
                    return "No expenses found"
                
                result = "Recent expenses:\n"
                for exp in expenses:
                    result += f"- Rs.{exp['amount']:.2f} on {exp['category']} ({exp.get('merchant', 'N/A')})\n"
                return result
            else:
                return f"Failed to get expenses: HTTP {response.status_code}"
        except Exception as e:
            return f"Error getting expenses: {str(e)}"
    
    def set_budget(self, category, amount):
        """Set budget for category"""
        data = {
            "category": category,
            "amount": amount,
            "period": "monthly",
            "user_id": self.user_id
        }
        
        try:
            response = requests.post(f"{self.base_url}/set-budget", json=data, timeout=10)
            if response.status_code == 200:
                return f"Budget set: Rs.{amount} for {category}"
            else:
                return f"Failed to set budget: HTTP {response.status_code}"
        except Exception as e:
            return f"Error setting budget: {str(e)}"
    
    def register_device(self):
        """Register demo device"""
        data = {
            "device_name": "Quick Demo Phone",
            "device_type": "Android",
            "phone_number": "+91-1234567890",
            "user_id": self.user_id
        }
        
        try:
            response = requests.post(f"{self.base_url}/register-device", json=data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                self.device_id = result.get("device_id")
                return f"Device registered: {self.device_id}"
            else:
                return f"Failed to register device: HTTP {response.status_code}"
        except Exception as e:
            return f"Error registering device: {str(e)}"
    
    def send_sample_sms(self):
        """Send a sample banking SMS"""
        if not self.device_id:
            reg_result = self.register_device()
            print(f"Device registration: {reg_result}")
        
        sms_data = {
            "sender": "DEMO-BANK",
            "message": "Alert: Rs 1500.00 debited from A/c **1234 on 01-Dec-24 to SWIGGY BANGALORE via UPI. Avl Bal: Rs 25,000.50",
            "timestamp": datetime.now().isoformat(),
            "device_id": self.device_id,
            "is_banking_sms": True
        }
        
        try:
            response = requests.post(f"{self.base_url}/sms-webhook", json=sms_data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                if result.get("expense_created"):
                    return f"SMS processed - Expense created: Rs.{result.get('amount', 0)} for {result.get('category', 'unknown')}"
                else:
                    return "SMS received but no expense created"
            else:
                return f"Failed to process SMS: HTTP {response.status_code}"
        except Exception as e:
            return f"Error processing SMS: {str(e)}"

def run_quick_demo():
    """Run quick demonstration"""
    print("Financial Assistant - Quick Demo")
    print("=" * 40)
    
    demo = QuickDemo()
    
    # Check API availability
    print("Checking API connection...")
    if not demo.check_api():
        print("ERROR: API is not running!")
        print("Please start the main application first:")
        print("python main.py")
        return
    
    print("API is running!")
    print()
    
    # Test basic functionality
    print("1. Testing Chat Interface:")
    print("-" * 25)
    
    # Chat tests
    chat_messages = [
        "Hello, what can you help me with?",
        "I spent 500 rupees on food today",
        "Show me my expenses",
        "Set 10000 rupees monthly budget for food"
    ]
    
    for msg in chat_messages:
        print(f"User: {msg}")
        response = demo.chat(msg)
        print(f"Bot: {response[:100]}{'...' if len(response) > 100 else ''}")
        print()
        time.sleep(1)  # Small delay between requests
    
    print("2. Testing Direct API Calls:")
    print("-" * 30)
    
    # Direct API tests
    print("Adding sample expenses...")
    expenses = [
        (250, "food", "Lunch at restaurant"),
        (150, "transport", "Uber ride"),
        (1200, "shopping", "Online purchase")
    ]
    
    for amount, category, desc in expenses:
        result = demo.add_expense(amount, category, desc)
        print(f"- {result}")
    
    print("\nSetting sample budgets...")
    budgets = [
        ("food", 5000),
        ("transport", 3000),
        ("shopping", 8000)
    ]
    
    for category, amount in budgets:
        result = demo.set_budget(category, amount)
        print(f"- {result}")
    
    print("\n3. Testing SMS Processing:")
    print("-" * 26)
    
    sms_result = demo.send_sample_sms()
    print(f"SMS Test: {sms_result}")
    
    print("\n4. Viewing Results:")
    print("-" * 18)
    
    expenses_list = demo.get_expenses()
    print(expenses_list)
    
    print("\n5. Interactive Chat Mode:")
    print("-" * 25)
    print("Type messages to chat with the assistant")
    print("Type 'quit' to exit")
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
            
            if user_input:
                response = demo.chat(user_input)
                print(f"Assistant: {response}")
                print()
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Demo completed!")

def run_developer_tests():
    """Run simplified developer tests"""
    print("Financial Assistant - Developer Tests")
    print("=" * 40)
    
    demo = QuickDemo()
    
    if not demo.check_api():
        print("ERROR: API not available!")
        return
    
    test_results = {}
    
    # Test 1: Basic API Health
    print("Test 1: API Health Check")
    try:
        response = requests.get(f"{demo.base_url}/health")
        test_results["health"] = response.status_code == 200
        print(f"Result: {'PASS' if test_results['health'] else 'FAIL'}")
    except:
        test_results["health"] = False
        print("Result: FAIL")
    
    # Test 2: Chat Functionality
    print("\nTest 2: Chat Interface")
    try:
        response = demo.chat("Hello test")
        test_results["chat"] = len(response) > 0
        print(f"Result: {'PASS' if test_results['chat'] else 'FAIL'}")
    except:
        test_results["chat"] = False
        print("Result: FAIL")
    
    # Test 3: Expense Management
    print("\nTest 3: Expense Management")
    try:
        add_result = demo.add_expense(100, "food", "Test expense")
        get_result = demo.get_expenses()
        test_results["expenses"] = "successfully" in add_result and "expenses" in get_result
        print(f"Result: {'PASS' if test_results['expenses'] else 'FAIL'}")
    except:
        test_results["expenses"] = False
        print("Result: FAIL")
    
    # Test 4: Budget Management
    print("\nTest 4: Budget Management")
    try:
        budget_result = demo.set_budget("food", 5000)
        test_results["budgets"] = "Budget set" in budget_result
        print(f"Result: {'PASS' if test_results['budgets'] else 'FAIL'}")
    except:
        test_results["budgets"] = False
        print("Result: FAIL")
    
    # Test 5: SMS Processing
    print("\nTest 5: SMS Processing")
    try:
        sms_result = demo.send_sample_sms()
        test_results["sms"] = "processed" in sms_result or "registered" in sms_result
        print(f"Result: {'PASS' if test_results['sms'] else 'FAIL'}")
    except:
        test_results["sms"] = False
        print("Result: FAIL")
    
    # Summary
    passed = sum(test_results.values())
    total = len(test_results)
    
    print(f"\nTest Summary: {passed}/{total} tests passed")
    print("=" * 40)
    
    for test, result in test_results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test.capitalize()}: {status}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_developer_tests()
    else:
        run_quick_demo()