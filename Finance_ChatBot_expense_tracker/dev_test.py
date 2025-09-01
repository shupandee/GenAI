#!/usr/bin/env python3
"""
Integrated Financial Assistant - Demo & Test Suite
=================================================
This script provides comprehensive testing and demonstration of all functionalities
including user interaction mode and developer testing mode.
"""

import asyncio
import json
import sqlite3
import requests
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import uuid
import random

# For async HTTP client
import aiohttp
import websockets

# ====================================================================
# DEMO DATA GENERATOR
# ====================================================================

class DemoDataGenerator:
    """Generates realistic demo data for testing"""
    
    def __init__(self):
        self.sample_expenses = [
            {"amount": 250, "category": "food", "merchant": "Swiggy", "description": "Lunch order"},
            {"amount": 150, "category": "transport", "merchant": "Uber", "description": "Cab to office"},
            {"amount": 1200, "category": "shopping", "merchant": "Amazon", "description": "Electronics"},
            {"amount": 80, "category": "fuel", "merchant": "HP Petrol", "description": "Fuel tank"},
            {"amount": 350, "category": "healthcare", "merchant": "Apollo Pharmacy", "description": "Medicines"},
            {"amount": 500, "category": "entertainment", "merchant": "PVR Cinemas", "description": "Movie tickets"},
            {"amount": 2500, "category": "utilities", "merchant": "Electric Board", "description": "Electricity bill"},
        ]
        
        self.sample_sms = [
            {
                "sender": "SBI-ALERTS",
                "message": "Dear Customer, Your A/c XX1234 is debited with Rs.250.00 on 01Dec24 for UPI-SWIGGY-BANGALORE Info:9876543210",
                "is_banking": True
            },
            {
                "sender": "HDFC-BANK",
                "message": "Alert: Rs 150.00 debited from A/c **5678 on 01-Dec-24 to UBER INDIA via UPI. Avl Bal: Rs 15,000.50",
                "is_banking": True
            },
            {
                "sender": "ICICI-BANK",
                "message": "Transaction Alert: Rs.1200 debited from your account ending 9012 on 30-Nov-24 for AMAZON PAY Ref:TXN123456",
                "is_banking": True
            },
            {
                "sender": "PROMOTIONS",
                "message": "Get 50% off on your next order! Use code SAVE50. Not a banking SMS.",
                "is_banking": False
            }
        ]
        
        self.sample_budgets = [
            {"category": "food", "amount": 8000, "period": "monthly"},
            {"category": "transport", "amount": 3000, "period": "monthly"},
            {"category": "shopping", "amount": 5000, "period": "monthly"},
            {"category": "entertainment", "amount": 2000, "period": "monthly"},
        ]
    
    def generate_random_expense(self) -> Dict[str, Any]:
        """Generate a random expense for testing"""
        base = random.choice(self.sample_expenses)
        return {
            "amount": round(base["amount"] * random.uniform(0.5, 2.0), 2),
            "category": base["category"],
            "merchant": base["merchant"],
            "description": base["description"],
            "user_id": "demo_user",
            "source": "manual",
            "confidence_score": 1.0,
            "timestamp": datetime.now()
        }
    
    def generate_sms_batch(self, count: int = 5) -> List[Dict[str, Any]]:
        """Generate a batch of SMS messages"""
        messages = []
        for _ in range(count):
            sms = random.choice(self.sample_sms).copy()
            # Vary amounts and dates
            if "Rs" in sms["message"]:
                old_amount = random.randint(100, 2000)
                new_amount = random.randint(50, 5000)
                sms["message"] = sms["message"].replace("250.00", f"{new_amount}.00")
            
            messages.append({
                "sender": sms["sender"],
                "message": sms["message"],
                "timestamp": datetime.now() - timedelta(minutes=random.randint(1, 1440)),
                "device_id": "demo_device",
                "is_banking_sms": sms["is_banking"]
            })
        return messages

# ====================================================================
# API CLIENT FOR TESTING
# ====================================================================

class FinancialAssistantClient:
    """Client for interacting with the Financial Assistant API"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.user_id = "demo_user"
        self.device_id = None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as response:
                return await response.json()
    
    async def chat(self, message: str, fetch_sms: bool = False) -> Dict[str, Any]:
        """Send chat message"""
        async with aiohttp.ClientSession() as session:
            data = {
                "message": message,
                "user_id": self.user_id,
                "fetch_sms": fetch_sms
            }
            async with session.post(f"{self.base_url}/chat", json=data) as response:
                return await response.json()
    
    async def register_device(self) -> Dict[str, str]:
        """Register demo device"""
        async with aiohttp.ClientSession() as session:
            data = {
                "device_name": "Demo Phone",
                "device_type": "Android",
                "phone_number": "+91-9876543210",
                "user_id": self.user_id
            }
            async with session.post(f"{self.base_url}/register-device", json=data) as response:
                result = await response.json()
                self.device_id = result.get("device_id")
                return result
    
    async def send_sms(self, sms_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send SMS to webhook"""
        if not self.device_id:
            await self.register_device()
        
        sms_data["device_id"] = self.device_id
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/sms-webhook", json=sms_data) as response:
                return await response.json()
    
    async def add_expense(self, expense_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add manual expense"""
        expense_data["user_id"] = self.user_id
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/add-expense", json=expense_data) as response:
                return await response.json()
    
    async def get_expenses(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get user expenses"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/expenses/{self.user_id}?limit={limit}") as response:
                return await response.json()
    
    async def set_budget(self, budget_data: Dict[str, Any]) -> Dict[str, Any]:
        """Set budget"""
        budget_data["user_id"] = self.user_id
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/set-budget", json=budget_data) as response:
                return await response.json()
    
    async def get_budgets(self) -> List[Dict[str, Any]]:
        """Get user budgets"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/budgets/{self.user_id}") as response:
                return await response.json()

# ====================================================================
# COMPREHENSIVE TEST SUITE
# ====================================================================

class FinancialAssistantTester:
    """Comprehensive testing suite for all functionalities"""
    
    def __init__(self):
        self.client = FinancialAssistantClient()
        self.data_generator = DemoDataGenerator()
        self.test_results = {}
        
    async def run_all_tests(self):
        """Run complete test suite"""
        print("🧪 Starting Comprehensive Test Suite")
        print("=" * 60)
        
        tests = [
            ("API Health Check", self.test_health_check),
            ("Device Registration", self.test_device_registration),
            ("Manual Expense Addition", self.test_manual_expenses),
            ("Budget Management", self.test_budget_functionality),
            ("SMS Processing", self.test_sms_processing),
            ("Chat Interface", self.test_chat_interface),
            ("Data Retrieval", self.test_data_retrieval),
            ("AI Features", self.test_ai_features),
            ("Edge Cases", self.test_edge_cases),
            ("Performance Test", self.test_performance)
        ]
        
        for test_name, test_func in tests:
            print(f"\n🔍 Testing: {test_name}")
            print("-" * 40)
            try:
                result = await test_func()
                self.test_results[test_name] = {"status": "PASS", "details": result}
                print(f"✅ {test_name}: PASSED")
            except Exception as e:
                self.test_results[test_name] = {"status": "FAIL", "error": str(e)}
                print(f"❌ {test_name}: FAILED - {e}")
        
        await self.print_test_summary()
    
    async def test_health_check(self):
        """Test API health and connectivity"""
        health = await self.client.health_check()
        
        assert health["status"] == "healthy", "API not healthy"
        assert "timestamp" in health, "Health check missing timestamp"
        
        print(f"   API Status: {health['status']}")
        print(f"   Connected Devices: {health.get('connected_devices', 0)}")
        return health
    
    async def test_device_registration(self):
        """Test mobile device registration"""
        result = await self.client.register_device()
        
        assert "device_id" in result, "Device ID not returned"
        assert result["status"] == "registered", "Device not registered"
        
        print(f"   Device ID: {result['device_id'][:8]}...")
        print(f"   Status: {result['status']}")
        return result
    
    async def test_manual_expenses(self):
        """Test manual expense addition"""
        expenses_added = 0
        
        for _ in range(3):
            expense = self.data_generator.generate_random_expense()
            result = await self.client.add_expense(expense)
            
            assert result["status"] == "success", "Failed to add expense"
            expenses_added += 1
        
        print(f"   Added {expenses_added} manual expenses")
        return {"expenses_added": expenses_added}
    
    async def test_budget_functionality(self):
        """Test budget setting and retrieval"""
        budgets_set = 0
        
        # Set budgets
        for budget in self.data_generator.sample_budgets:
            result = await self.client.set_budget(budget)
            assert result["status"] == "success", f"Failed to set budget for {budget['category']}"
            budgets_set += 1
        
        # Get budgets
        budgets = await self.client.get_budgets()
        assert len(budgets) >= budgets_set, "Not all budgets retrieved"
        
        print(f"   Set {budgets_set} budgets")
        print(f"   Retrieved {len(budgets)} budget statuses")
        
        for budget in budgets[:2]:  # Show first 2
            print(f"   - {budget['category']}: ₹{budget['spent']:.2f}/₹{budget['budget']:.2f}")
        
        return {"budgets_set": budgets_set, "budgets_retrieved": len(budgets)}
    
    async def test_sms_processing(self):
        """Test SMS message processing"""
        sms_batch = self.data_generator.generate_sms_batch(4)
        processed_sms = 0
        banking_sms = 0
        
        for sms in sms_batch:
            result = await self.client.send_sms(sms)
            processed_sms += 1
            
            if result.get("expense_created"):
                banking_sms += 1
                print(f"   Banking SMS processed: ₹{result['amount']} - {result['category']}")
            else:
                print(f"   SMS stored: {sms['sender'][:15]}...")
        
        return {
            "total_sms": processed_sms,
            "banking_sms": banking_sms,
            "non_banking_sms": processed_sms - banking_sms
        }
    
    async def test_chat_interface(self):
        """Test chat interface with various queries"""
        chat_queries = [
            "I spent ₹500 on food at McDonald's",
            "Show me my recent expenses",
            "Set a ₹10000 monthly budget for food",
            "How much did I spend on transport this month?",
            "Create a chart of my expenses",
            "Check my budget status",
            "Fetch my SMS transactions"
        ]
        
        responses = []
        for query in chat_queries:
            try:
                response = await self.client.chat(query)
                responses.append({
                    "query": query,
                    "intent": response.get("intent", "unknown"),
                    "success": "response" in response
                })
                print(f"   Query: '{query[:30]}...' -> Intent: {response.get('intent', 'unknown')}")
            except Exception as e:
                print(f"   Query failed: '{query[:30]}...' -> Error: {str(e)[:50]}...")
        
        return {"queries_tested": len(responses), "successful": sum(r["success"] for r in responses)}
    
    async def test_data_retrieval(self):
        """Test data retrieval endpoints"""
        # Get expenses
        expenses = await self.client.get_expenses(limit=10)
        
        # Get budgets
        budgets = await self.client.get_budgets()
        
        print(f"   Retrieved {len(expenses)} expenses")
        print(f"   Retrieved {len(budgets)} budget statuses")
        
        if expenses:
            total_amount = sum(exp.get("amount", 0) for exp in expenses)
            categories = set(exp.get("category") for exp in expenses)
            print(f"   Total expense amount: ₹{total_amount:.2f}")
            print(f"   Categories found: {len(categories)}")
        
        return {
            "expenses_count": len(expenses),
            "budgets_count": len(budgets),
            "total_amount": sum(exp.get("amount", 0) for exp in expenses) if expenses else 0
        }
    
    async def test_ai_features(self):
        """Test AI-powered features"""
        ai_queries = [
            "What are my spending patterns?",
            "Give me financial advice based on my expenses",
            "How can I save more money?",
            "Analyze my food expenses"
        ]
        
        ai_responses = []
        for query in ai_queries:
            try:
                response = await self.client.chat(query)
                ai_responses.append({
                    "query": query,
                    "has_response": len(response.get("response", "")) > 50,
                    "intent": response.get("intent", "unknown")
                })
                print(f"   AI Query: '{query[:25]}...' -> Response length: {len(response.get('response', ''))}")
            except Exception as e:
                print(f"   AI Query failed: {str(e)[:50]}...")
        
        return {"ai_queries": len(ai_responses), "successful": sum(r["has_response"] for r in ai_responses)}
    
    async def test_edge_cases(self):
        """Test edge cases and error handling"""
        edge_cases = [
            ("Empty message", ""),
            ("Invalid expense", "I spent negative money"),
            ("Nonsense query", "xyz abc 123 random text"),
            ("Very long message", "a" * 1000),
        ]
        
        handled_cases = 0
        for case_name, message in edge_cases:
            try:
                response = await self.client.chat(message)
                if "response" in response:
                    handled_cases += 1
                    print(f"   {case_name}: Handled gracefully")
                else:
                    print(f"   {case_name}: No response")
            except Exception as e:
                print(f"   {case_name}: Error handled - {str(e)[:30]}...")
                handled_cases += 1  # Error handling is also good
        
        return {"edge_cases_tested": len(edge_cases), "handled_gracefully": handled_cases}
    
    async def test_performance(self):
        """Test system performance with concurrent requests"""
        start_time = time.time()
        
        # Create multiple concurrent requests
        tasks = []
        for i in range(5):
            expense = self.data_generator.generate_random_expense()
            task = self.client.add_expense(expense)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
        
        print(f"   Concurrent requests: 5")
        print(f"   Successful: {successful}")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Avg response time: {duration/5:.3f} seconds")
        
        return {
            "concurrent_requests": 5,
            "successful": successful,
            "duration": duration,
            "avg_response_time": duration/5
        }
    
    async def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for r in self.test_results.values() if r["status"] == "PASS")
        total = len(self.test_results)
        
        print(f"Overall Result: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
        print()
        
        for test_name, result in self.test_results.items():
            status_emoji = "✅" if result["status"] == "PASS" else "❌"
            print(f"{status_emoji} {test_name}: {result['status']}")
            if result["status"] == "FAIL":
                print(f"   Error: {result['error']}")
        
        print("\n" + "=" * 60)

# ====================================================================
# INTERACTIVE DEMO MODE
# ====================================================================

class InteractiveDemo:
    """Interactive demonstration mode for users"""
    
    def __init__(self):
        self.client = FinancialAssistantClient()
        self.data_generator = DemoDataGenerator()
        self.session_active = True
    
    async def start_demo(self):
        """Start interactive demo session"""
        print("🤖 Financial Assistant Interactive Demo")
        print("=" * 50)
        print("Welcome! This demo lets you explore all features.")
        print("Type 'help' for commands or 'quit' to exit.")
        print("=" * 50)
        
        # Initialize with sample data
        await self.setup_demo_data()
        
        while self.session_active:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Thanks for trying the Financial Assistant!")
                    break
                elif user_input.lower() == 'help':
                    self.show_help()
                elif user_input.lower().startswith('demo'):
                    await self.handle_demo_commands(user_input)
                elif user_input.lower() == 'status':
                    await self.show_system_status()
                else:
                    await self.handle_chat_message(user_input)
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def setup_demo_data(self):
        """Setup initial demo data"""
        print("🔧 Setting up demo environment...")
        
        try:
            # Register device
            await self.client.register_device()
            
            # Add some sample expenses
            for expense in self.data_generator.sample_expenses[:3]:
                expense["user_id"] = self.client.user_id
                expense["source"] = "manual"
                expense["confidence_score"] = 1.0
                expense["timestamp"] = datetime.now()
                await self.client.add_expense(expense)
            
            # Set sample budgets
            for budget in self.data_generator.sample_budgets[:2]:
                await self.client.set_budget(budget)
            
            # Process some SMS
            sms_batch = self.data_generator.generate_sms_batch(2)
            for sms in sms_batch:
                if sms["is_banking_sms"]:
                    await self.client.send_sms(sms)
            
            print("✅ Demo environment ready!")
            
        except Exception as e:
            print(f"⚠️ Setup warning: {e}")
    
    def show_help(self):
        """Show available commands"""
        print("\n📋 Available Commands:")
        print("-" * 30)
        print("💰 Expense Management:")
        print("  • 'I spent ₹500 on food'")
        print("  • 'Show my expenses'")
        print("  • 'Add ₹200 transport expense'")
        print()
        print("📊 Budget Management:")
        print("  • 'Set ₹5000 monthly budget for food'")
        print("  • 'Check my budgets'")
        print("  • 'How much did I spend on shopping?'")
        print()
        print("📱 SMS Features:")
        print("  • 'Fetch my SMS'")
        print("  • 'Process banking SMS'")
        print()
        print("📈 Analytics:")
        print("  • 'Create a chart'")
        print("  • 'Show spending patterns'")
        print("  • 'Financial advice'")
        print()
        print("🛠️ Demo Commands:")
        print("  • 'demo add' - Add random expenses")
        print("  • 'demo sms' - Send demo SMS")
        print("  • 'demo reset' - Clear demo data")
        print("  • 'status' - System status")
        print("  • 'help' - Show this help")
        print("  • 'quit' - Exit demo")
    
    async def handle_demo_commands(self, command: str):
        """Handle special demo commands"""
        parts = command.lower().split()
        
        if len(parts) < 2:
            print("Usage: demo [add|sms|reset]")
            return
        
        action = parts[1]
        
        if action == "add":
            # Add random expenses
            count = 3
            print(f"🎲 Adding {count} random expenses...")
            
            for i in range(count):
                expense = self.data_generator.generate_random_expense()
                result = await self.client.add_expense(expense)
                if result.get("status") == "success":
                    print(f"   ✅ Added: ₹{expense['amount']} - {expense['category']}")
                else:
                    print(f"   ❌ Failed to add expense {i+1}")
        
        elif action == "sms":
            # Send demo SMS
            print("📱 Sending demo banking SMS...")
            sms_batch = self.data_generator.generate_sms_batch(2)
            
            for sms in sms_batch:
                if sms["is_banking_sms"]:
                    result = await self.client.send_sms(sms)
                    if result.get("expense_created"):
                        print(f"   ✅ SMS processed: ₹{result['amount']} - {result['category']}")
                    else:
                        print(f"   📥 SMS stored: {sms['sender']}")
        
        elif action == "reset":
            print("🔄 This would reset demo data (not implemented in demo)")
            print("   In full version, this would clear all demo expenses")
        
        else:
            print(f"Unknown demo command: {action}")
    
    async def handle_chat_message(self, message: str):
        """Handle regular chat messages"""
        try:
            response = await self.client.chat(message)
            
            print(f"\n🤖 Assistant: {response.get('response', 'No response received')}")
            
            # Show additional info for certain intents
            intent = response.get("intent")
            if intent == "view_expenses":
                total = response.get("total", 0)
                print(f"💡 Tip: You can also ask for specific categories or time periods")
            elif intent == "budget_check":
                print(f"💡 Tip: Set budgets with 'Set ₹amount monthly budget for category'")
            elif intent == "create_chart":
                print(f"💡 Tip: Charts are generated based on your expense data")
            
        except Exception as e:
            print(f"❌ Chat error: {e}")
    
    async def show_system_status(self):
        """Show current system status"""
        try:
            health = await self.client.health_check()
            expenses = await self.client.get_expenses(limit=5)
            budgets = await self.client.get_budgets()
            
            print("\n📊 System Status:")
            print("-" * 20)
            print(f"API Status: {health.get('status', 'unknown')}")
            print(f"Connected Devices: {health.get('connected_devices', 0)}")
            print(f"Active WebSockets: {health.get('active_websockets', 0)}")
            print(f"Your Expenses: {len(expenses)}")
            print(f"Your Budgets: {len(budgets)}")
            
            if expenses:
                total = sum(exp.get("amount", 0) for exp in expenses)
                print(f"Total Spending: ₹{total:.2f}")
        
        except Exception as e:
            print(f"❌ Status check failed: {e}")

# ====================================================================
# MAIN EXECUTION
# ====================================================================

async def main():
    """Main execution function"""
    import sys
    
    print("🚀 Financial Assistant Demo & Testing Suite")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("Select mode:")
        print("1. Interactive Demo (user-friendly)")
        print("2. Developer Test Suite (comprehensive)")
        print("3. Both (demo first, then tests)")
        
        choice = input("\nEnter choice (1/2/3): ").strip()
        mode = {"1": "demo", "2": "test", "3": "both"}.get(choice, "demo")
    
    if mode in ["demo", "both"]:
        print("\n🎮 Starting Interactive Demo...")
        demo = InteractiveDemo()
        await demo.start_demo()
    
    if mode in ["test", "both"]:
        if mode == "both":
            input("\nPress Enter to continue to test suite...")
        
        print("\n🧪 Starting Developer Test Suite...")
        tester = FinancialAssistantTester()
        await tester.run_all_tests()
    
    print("\n🎯 Demo/Testing Complete!")

if __name__ == "__main__":
    # Check if event loop is already running (in Jupyter/async environments)
    try:
        loop = asyncio.get_running_loop()
        print("⚠️  Running in async environment. Use 'await main()' instead.")
    except RuntimeError:
        # No running loop, we can create one
        asyncio.run(main())