"""
Integrated Financial Assistant with SMS Auto-Fetch Capabilities
================================================================
This system combines:
1. AI-powered chatbot for financial management
2. Real-time SMS expense tracking
3. Automatic expense categorization
4. Budget management and visualizations
"""

import os
import json
import sqlite3
import asyncio
import logging
import uuid
import re
import hashlib
import base64
import io
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import threading
import time

# FastAPI and WebSocket imports
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# LangChain imports for AI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ====================================================================
# SHARED ENUMS AND MODELS
# ====================================================================

class ExpenseCategory(str, Enum):
    FOOD = "food"
    TRANSPORT = "transport"
    SHOPPING = "shopping"
    UTILITIES = "utilities"
    ENTERTAINMENT = "entertainment"
    HEALTHCARE = "healthcare"
    FUEL = "fuel"
    EDUCATION = "education"
    HOUSING = "housing"
    INSURANCE = "insurance"
    SAVINGS = "savings"
    INVESTMENT = "investment"
    ATM_WITHDRAWAL = "atm_withdrawal"
    OTHER = "other"

class DataSource(str, Enum):
    MANUAL = "manual"
    SMS = "sms"
    MOBILE_SMS = "mobile_sms"
    CHATBOT = "chatbot"
    API = "api"

class DeviceStatus(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    PENDING = "pending"
    ERROR = "error"

# ====================================================================
# PYDANTIC MODELS
# ====================================================================

class UserPermission(BaseModel):
    user_id: str
    allow_sms_fetch: bool = False
    allow_auto_categorize: bool = True
    allow_budget_alerts: bool = True
    device_id: Optional[str] = None

class ChatMessage(BaseModel):
    message: str
    user_id: str = "default_user"
    fetch_sms: bool = False  # User can request SMS fetch in chat

class SMSMessage(BaseModel):
    sender: str
    message: str
    timestamp: datetime
    device_id: str
    message_id: Optional[str] = None
    is_banking_sms: bool = False

class DeviceRegistration(BaseModel):
    device_name: str
    device_type: str
    phone_number: Optional[str] = None
    user_id: str = "default_user"

class ExpenseEntry(BaseModel):
    amount: float
    category: ExpenseCategory
    description: str
    merchant: str
    timestamp: datetime
    source: DataSource
    confidence_score: float
    user_id: str
    device_id: Optional[str] = None

class BudgetSet(BaseModel):
    category: str
    amount: float
    period: str = "monthly"
    user_id: str = "default_user"

# ====================================================================
# UNIFIED DATABASE HANDLER
# ====================================================================

class UnifiedFinancialDatabase:
    """Unified database handler for expenses, SMS, budgets, and permissions"""
    
    def __init__(self, db_path: str = "unified_financial.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Initialize all required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table with permissions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                allow_sms_fetch BOOLEAN DEFAULT 0,
                allow_auto_categorize BOOLEAN DEFAULT 1,
                allow_budget_alerts BOOLEAN DEFAULT 1,
                primary_device_id TEXT
            )
        ''')
        
        # Unified expenses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS expenses (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                amount REAL NOT NULL,
                category TEXT NOT NULL,
                description TEXT,
                merchant TEXT,
                date TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence_score REAL DEFAULT 1.0,
                device_id TEXT,
                sms_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Mobile devices table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mobile_devices (
                device_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                device_name TEXT,
                device_type TEXT,
                phone_number TEXT,
                registration_time TIMESTAMP,
                last_active TIMESTAMP,
                status TEXT,
                total_messages_sent INTEGER DEFAULT 0,
                banking_messages_detected INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # SMS messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sms_messages (
                id TEXT PRIMARY KEY,
                device_id TEXT,
                user_id TEXT,
                sender TEXT,
                message TEXT,
                timestamp TIMESTAMP,
                is_banking_sms BOOLEAN,
                processed BOOLEAN DEFAULT 0,
                expense_id TEXT,
                FOREIGN KEY (device_id) REFERENCES mobile_devices (device_id),
                FOREIGN KEY (expense_id) REFERENCES expenses (id)
            )
        ''')
        
        # Budgets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS budgets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                category TEXT NOT NULL,
                amount REAL NOT NULL,
                period TEXT DEFAULT 'monthly',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, category, period),
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Categories table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                keywords TEXT
            )
        ''')
        
        # Insert default categories with keywords
        default_categories = [
            ('food', 'Food and dining', 'swiggy,zomato,restaurant,lunch,dinner,breakfast,cafe,pizza,burger'),
            ('transport', 'Transportation', 'uber,ola,rapido,cab,taxi,bus,metro,train,flight'),
            ('shopping', 'Shopping and retail', 'amazon,flipkart,myntra,mall,shopping,store'),
            ('utilities', 'Bills and utilities', 'electricity,water,gas,internet,phone,mobile,recharge'),
            ('entertainment', 'Entertainment', 'movie,netflix,spotify,game,concert,show'),
            ('healthcare', 'Medical and health', 'hospital,clinic,pharmacy,medicine,doctor,medical'),
            ('fuel', 'Fuel and gas', 'petrol,diesel,fuel,gas,shell,hp,bpcl'),
            ('education', 'Educational expenses', 'school,college,course,book,tuition'),
            ('housing', 'Housing and rent', 'rent,mortgage,maintenance,repair'),
            ('insurance', 'Insurance premiums', 'insurance,premium,policy'),
            ('savings', 'Savings and deposits', 'savings,deposit,fd,rd'),
            ('investment', 'Investments', 'investment,stock,mutual,fund,share'),
            ('atm_withdrawal', 'ATM withdrawals', 'atm,withdrawal,cash'),
            ('other', 'Other expenses', '')
        ]
        
        cursor.executemany('''
            INSERT OR IGNORE INTO categories (name, description, keywords) VALUES (?, ?, ?)
        ''', default_categories)
        
        conn.commit()
        conn.close()
    
    def get_or_create_user(self, user_id: str) -> Dict[str, Any]:
        """Get or create user with permissions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
        user = cursor.fetchone()
        
        if not user:
            cursor.execute('''
                INSERT INTO users (user_id) VALUES (?)
            ''', (user_id,))
            conn.commit()
            cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
            user = cursor.fetchone()
        
        conn.close()
        
        return {
            'user_id': user[0],
            'created_at': user[1],
            'last_active': user[2],
            'allow_sms_fetch': bool(user[3]),
            'allow_auto_categorize': bool(user[4]),
            'allow_budget_alerts': bool(user[5]),
            'primary_device_id': user[6]
        }
    
    def update_user_permissions(self, user_id: str, permissions: Dict[str, Any]):
        """Update user permissions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users SET 
                allow_sms_fetch = ?,
                allow_auto_categorize = ?,
                allow_budget_alerts = ?,
                primary_device_id = ?,
                last_active = CURRENT_TIMESTAMP
            WHERE user_id = ?
        ''', (
            permissions.get('allow_sms_fetch', False),
            permissions.get('allow_auto_categorize', True),
            permissions.get('allow_budget_alerts', True),
            permissions.get('primary_device_id'),
            user_id
        ))
        
        conn.commit()
        conn.close()
    
    def add_expense(self, expense_data: Dict[str, Any]) -> str:
        """Add expense to unified database"""
        expense_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO expenses 
            (id, user_id, amount, category, description, merchant, date, source, 
             confidence_score, device_id, sms_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            expense_id,
            expense_data['user_id'],
            expense_data['amount'],
            expense_data['category'],
            expense_data.get('description', ''),
            expense_data.get('merchant', ''),
            expense_data.get('date', datetime.now().strftime('%Y-%m-%d')),
            expense_data.get('source', 'manual'),
            expense_data.get('confidence_score', 1.0),
            expense_data.get('device_id'),
            expense_data.get('sms_id')
        ))
        
        conn.commit()
        conn.close()
        
        return expense_id
    
    def get_unprocessed_sms(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get unprocessed SMS messages for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT s.* FROM sms_messages s
            JOIN mobile_devices d ON s.device_id = d.device_id
            WHERE d.user_id = ? AND s.processed = 0 AND s.is_banking_sms = 1
            ORDER BY s.timestamp DESC
            LIMIT ?
        ''', (user_id, limit))
        
        messages = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': msg[0],
                'device_id': msg[1],
                'user_id': msg[2],
                'sender': msg[3],
                'message': msg[4],
                'timestamp': msg[5],
                'is_banking_sms': bool(msg[6])
            }
            for msg in messages
        ]
    
    def mark_sms_processed(self, sms_id: str, expense_id: str = None):
        """Mark SMS as processed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE sms_messages 
            SET processed = 1, expense_id = ?
            WHERE id = ?
        ''', (expense_id, sms_id))
        
        conn.commit()
        conn.close()
    
    def register_device(self, device_data: Dict[str, Any]) -> str:
        """Register a new mobile device"""
        device_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO mobile_devices 
            (device_id, user_id, device_name, device_type, phone_number, 
             registration_time, last_active, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            device_id,
            device_data['user_id'],
            device_data['device_name'],
            device_data['device_type'],
            device_data.get('phone_number'),
            datetime.now(),
            datetime.now(),
            'connected'
        ))
        
        conn.commit()
        conn.close()
        
        return device_id
    
    def store_sms_message(self, sms_data: Dict[str, Any]) -> str:
        """Store SMS message in database"""
        sms_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Determine if it's a banking SMS
        is_banking = self._is_banking_sms(sms_data['message'])
        
        cursor.execute('''
            INSERT INTO sms_messages 
            (id, device_id, user_id, sender, message, timestamp, is_banking_sms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            sms_id,
            sms_data['device_id'],
            sms_data['user_id'],
            sms_data['sender'],
            sms_data['message'],
            sms_data['timestamp'],
            is_banking
        ))
        
        conn.commit()
        conn.close()
        
        return sms_id
    
    def _is_banking_sms(self, message: str) -> bool:
        """Check if SMS is banking related"""
        banking_keywords = [
            'debited', 'credited', 'debit', 'credit', 'transaction', 'payment',
            'bank', 'account', 'atm', 'upi', 'neft', 'rtgs', 'imps',
            'balance', 'available', 'spent', 'paid'
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in banking_keywords)

# ====================================================================
# ENHANCED SMS PROCESSOR WITH AI
# ====================================================================

class IntelligentSMSProcessor:
    """SMS processor with AI-powered categorization"""
    
    def __init__(self, llm=None):
        self.llm = llm
        self.banking_patterns = {
            'debit_patterns': [
                r'Rs\.?(\d+(?:,\d+)*(?:\.\d{2})?)\s*(?:Dr|debited|debit)',
                r'(\d+(?:,\d+)*(?:\.\d{2})?)\s*(?:Dr|debited|debit)',
                r'debited.*?Rs\.?(\d+(?:,\d+)*(?:\.\d{2})?)',
                r'spent.*?Rs\.?(\d+(?:,\d+)*(?:\.\d{2})?)',
                r'paid.*?Rs\.?(\d+(?:,\d+)*(?:\.\d{2})?)'
            ]
        }
    
    def process_sms_with_ai(self, sms_text: str, sender: str) -> Optional[Dict[str, Any]]:
        """Process SMS using AI for better categorization"""
        
        # Extract amount using patterns
        amount = self._extract_amount(sms_text)
        if not amount:
            return None
        
        # Use AI for categorization if available
        if self.llm:
            prompt = f"""
            Analyze this banking SMS and extract:
            1. Category (food/transport/shopping/utilities/healthcare/fuel/education/other)
            2. Merchant name
            3. Brief description
            
            SMS from {sender}: "{sms_text}"
            
            Return as JSON: {{"category": "...", "merchant": "...", "description": "..."}}
            """
            
            try:
                response = self.llm.predict(prompt)
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    ai_data = json.loads(json_match.group())
                    return {
                        'amount': amount,
                        'category': ai_data.get('category', 'other'),
                        'merchant': ai_data.get('merchant', sender),
                        'description': ai_data.get('description', sms_text[:100]),
                        'confidence_score': 0.9
                    }
            except Exception as e:
                logger.error(f"AI processing failed: {e}")
        
        # Fallback to pattern-based extraction
        return {
            'amount': amount,
            'category': self._categorize_by_keywords(sms_text),
            'merchant': self._extract_merchant(sms_text) or sender,
            'description': sms_text[:100],
            'confidence_score': 0.7
        }
    
    def _extract_amount(self, message: str) -> Optional[float]:
        """Extract transaction amount from message"""
        for pattern in self.banking_patterns['debit_patterns']:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    return float(amount_str)
                except ValueError:
                    continue
        return None
    
    def _extract_merchant(self, message: str) -> Optional[str]:
        """Extract merchant name from message"""
        patterns = [
            r'(?:to|at)\s+([A-Z][A-Z0-9\s\-\.]{3,30})',
            r'(?:UPI-|UPI ID-)([a-zA-Z0-9@\.\-]+)',
            r'([a-zA-Z0-9]+@[a-zA-Z]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def _categorize_by_keywords(self, text: str) -> str:
        """Categorize based on keywords"""
        text_lower = text.lower()
        
        category_keywords = {
            'food': ['swiggy', 'zomato', 'food', 'restaurant', 'cafe'],
            'transport': ['uber', 'ola', 'rapido', 'cab', 'taxi'],
            'shopping': ['amazon', 'flipkart', 'myntra', 'shopping'],
            'fuel': ['petrol', 'diesel', 'fuel', 'hp', 'bpcl'],
            'healthcare': ['hospital', 'clinic', 'pharmacy', 'medical']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return 'other'

# ====================================================================
# INTEGRATED FINANCIAL ASSISTANT
# ====================================================================

class IntegratedFinancialAssistant:
    """Main assistant combining chatbot and SMS processing"""
    
    def __init__(self, model_provider: str = "gemini", api_key: str = None):
        # Initialize LLM
        if model_provider == "gemini":
            if api_key:
                os.environ["GOOGLE_API_KEY"] = "Gemni_API_Key_here"
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.1,
                convert_system_message_to_human=True
            )
        elif model_provider == "cohere":
            if api_key:
                os.environ["COHERE_API_KEY"] = "Your_COHERE_Api_key_here"
            self.llm = ChatCohere(
                model="command-r",
                temperature=0.1
            )
        
        # Initialize components
        self.db = UnifiedFinancialDatabase()
        self.sms_processor = IntelligentSMSProcessor(self.llm)
        self.user_memories = {}
        self.connected_devices = {}
        self.websocket_connections = {}
        
        # Chart generator
        plt.style.use('seaborn-v0_8')
    
    async def process_chat_message(self, message: str, user_id: str, fetch_sms: bool = False) -> Dict[str, Any]:
        """Process chat message with optional SMS fetching"""
        
        # Get user permissions
        user = self.db.get_or_create_user(user_id)
        
        # Check if user requested SMS fetch
        if fetch_sms or "fetch" in message.lower() or "sync" in message.lower():
            if user['allow_sms_fetch']:
                fetch_result = await self.fetch_and_process_sms(user_id)
                if fetch_result['processed_count'] > 0:
                    return {
                        'response': f"✅ Fetched and processed {fetch_result['processed_count']} SMS transactions!\n"
                                   f"Total amount: ₹{fetch_result['total_amount']:.2f}\n\n"
                                   f"Now, how can I help you with your finances?",
                        'intent': 'sms_fetch',
                        'fetch_result': fetch_result
                    }
            else:
                return {
                    'response': "📱 SMS fetching is not enabled. Would you like to enable it?\n"
                               "Reply with 'enable sms fetching' to allow automatic expense tracking from SMS.",
                    'intent': 'permission_request'
                }
        
        # Handle permission requests
        if "enable sms" in message.lower():
            self.db.update_user_permissions(user_id, {
                'allow_sms_fetch': True,
                'allow_auto_categorize': True,
                'allow_budget_alerts': True
            })
            return {
                'response': "✅ SMS fetching enabled! You can now:\n"
                           "• Say 'fetch my sms' to import banking transactions\n"
                           "• Register your device for real-time SMS tracking\n"
                           "• All SMS data will be automatically categorized",
                'intent': 'permission_granted'
            }
        
        # Regular chat processing
        intent = self._classify_intent(message)
        
        if intent == "add_expense":
            return await self._handle_add_expense(message, user_id)
        elif intent == "view_expenses":
            return await self._handle_view_expenses(user_id)
        elif intent == "category_analysis":
            return await self._handle_category_analysis(message, user_id)
        elif intent == "budget_set":
            return await self._handle_budget_set(message, user_id)
        elif intent == "budget_check":
            return await self._handle_budget_check(user_id)
        elif intent == "create_chart":
            return await self._handle_create_chart(message, user_id)
        else:
            return await self._handle_general_query(message, user_id)
    
    async def fetch_and_process_sms(self, user_id: str) -> Dict[str, Any]:
        """Fetch and process unprocessed SMS messages"""
        
        unprocessed = self.db.get_unprocessed_sms(user_id)
        processed_count = 0
        total_amount = 0
        expenses_created = []
        
        for sms in unprocessed:
            # Process with AI
            result = self.sms_processor.process_sms_with_ai(
                sms['message'], 
                sms['sender']
            )
            
            if result:
                # Create expense
                expense_id = self.db.add_expense({
                    'user_id': user_id,
                    'amount': result['amount'],
                    'category': result['category'],
                    'description': result['description'],
                    'merchant': result['merchant'],
                    'source': 'mobile_sms',
                    'confidence_score': result['confidence_score'],
                    'device_id': sms['device_id'],
                    'sms_id': sms['id']
                })
                
                # Mark SMS as processed
                self.db.mark_sms_processed(sms['id'], expense_id)
                
                processed_count += 1
                total_amount += result['amount']
                expenses_created.append({
                    'id': expense_id,
                    'amount': result['amount'],
                    'category': result['category'],
                    'merchant': result['merchant']
                })
        
        return {
            'processed_count': processed_count,
            'total_amount': total_amount,
            'expenses_created': expenses_created
        }
    
    def _classify_intent(self, text: str) -> str:
        """Classify user intent"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['spent', 'paid', 'bought', 'cost']):
            return 'add_expense'
        elif any(word in text_lower for word in ['show', 'view', 'list', 'see']) and 'expense' in text_lower:
            return 'view_expenses'
        elif 'budget' in text_lower and 'set' in text_lower:
            return 'budget_set'
        elif 'budget' in text_lower:
            return 'budget_check'
        elif any(word in text_lower for word in ['chart', 'graph', 'visualization', 'plot']):
            return 'create_chart'
        elif any(cat in text_lower for cat in ['food', 'transport', 'shopping']):
            return 'category_analysis'
        else:
            return 'general'
    
    async def _handle_add_expense(self, message: str, user_id: str) -> Dict[str, Any]:
        """Handle expense addition from chat"""
        
        # Use AI to extract expense details
        prompt = f"""
        Extract expense information from: "{message}"
        Return as JSON: {{"amount": number, "category": "...", "description": "..."}}
        Categories: food, transport, shopping, utilities, healthcare, fuel, education, other
        """
        
        try:
            response = self.llm.predict(prompt)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                expense_data = json.loads(json_match.group())
                
                expense_id = self.db.add_expense({
                    'user_id': user_id,
                    'amount': expense_data['amount'],
                    'category': expense_data.get('category', 'other'),
                    'description': expense_data.get('description', ''),
                    'source': 'chatbot'
                })
                
                return {
                    'response': f"✅ Expense added: ₹{expense_data['amount']} for {expense_data.get('category', 'other')}",
                    'intent': 'add_expense',
                    'expense_id': expense_id
                }
        except Exception as e:
            logger.error(f"Failed to add expense: {e}")
        
        return {
            'response': "I couldn't understand the expense details. Please try: 'I spent ₹500 on food'",
            'intent': 'add_expense_failed'
        }
    
    async def _handle_view_expenses(self, user_id: str) -> Dict[str, Any]:
        """Handle view expenses request"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT amount, category, merchant, date, source
            FROM expenses
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 10
        ''', (user_id,))
        
        expenses = cursor.fetchall()
        conn.close()
        
        if not expenses:
            return {
                'response': "No expenses found. Add some expenses or fetch from SMS!",
                'intent': 'view_expenses'
            }
        
        response = "📊 Your recent expenses:\n\n"
        total = 0
        
        for exp in expenses:
            amount, category, merchant, date, source = exp
            total += amount
            response += f"• ₹{amount:.2f} - {category} ({merchant or 'N/A'}) on {date}"
            if source == 'mobile_sms':
                response += " 📱"
            response += "\n"
        
        response += f"\n💰 Total: ₹{total:.2f}"
        
        return {
            'response': response,
            'intent': 'view_expenses',
            'total': total
        }
    
    async def _handle_category_analysis(self, message: str, user_id: str) -> Dict[str, Any]:
        """Handle category analysis request"""
        # Extract category from message
        categories = ['food', 'transport', 'shopping', 'utilities', 'healthcare', 'fuel']
        category = None
        
        for cat in categories:
            if cat in message.lower():
                category = cat
                break
        
        if not category:
            return {
                'response': "Please specify a category (food, transport, shopping, etc.)",
                'intent': 'category_analysis'
            }
        
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT SUM(amount) as total, COUNT(*) as count, AVG(amount) as avg
            FROM expenses
            WHERE user_id = ? AND category = ?
            AND date >= date('now', '-30 days')
        ''', (user_id, category))
        
        result = cursor.fetchone()
        conn.close()
        
        if result[0]:
            return {
                'response': f"📊 {category.title()} spending (last 30 days):\n"
                           f"• Total: ₹{result[0]:.2f}\n"
                           f"• Transactions: {result[1]}\n"
                           f"• Average: ₹{result[2]:.2f}",
                'intent': 'category_analysis',
                'data': {
                    'category': category,
                    'total': result[0],
                    'count': result[1],
                    'average': result[2]
                }
            }
        else:
            return {
                'response': f"No {category} expenses found in the last 30 days",
                'intent': 'category_analysis'
            }
    
    async def _handle_budget_set(self, message: str, user_id: str) -> Dict[str, Any]:
        """Handle budget setting"""
        # Extract budget details using AI
        prompt = f"""
        Extract budget information from: "{message}"
        Return as JSON: {{"category": "...", "amount": number, "period": "monthly/weekly/yearly"}}
        """
        
        try:
            response = self.llm.predict(prompt)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                budget_data = json.loads(json_match.group())
                
                conn = sqlite3.connect(self.db.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO budgets (user_id, category, amount, period)
                    VALUES (?, ?, ?, ?)
                ''', (user_id, budget_data['category'], budget_data['amount'], 
                      budget_data.get('period', 'monthly')))
                
                conn.commit()
                conn.close()
                
                return {
                    'response': f"✅ Budget set: ₹{budget_data['amount']} for {budget_data['category']} ({budget_data.get('period', 'monthly')})",
                    'intent': 'budget_set'
                }
        except Exception as e:
            logger.error(f"Failed to set budget: {e}")
        
        return {
            'response': "Please specify budget like: 'Set ₹5000 monthly budget for food'",
            'intent': 'budget_set_failed'
        }
    
    async def _handle_budget_check(self, user_id: str) -> Dict[str, Any]:
        """Check budget status"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        # Get all budgets for user
        cursor.execute('''
            SELECT category, amount, period FROM budgets 
            WHERE user_id = ?
        ''', (user_id,))
        budgets = cursor.fetchall()
        
        if not budgets:
            conn.close()
            return {
                'response': "No budgets set. Set a budget like: 'Set ₹5000 monthly budget for food'",
                'intent': 'budget_check'
            }
        
        response = "💰 Budget Status:\n\n"
        
        for budget in budgets:
            category, budget_amount, period = budget
            
            # Calculate period start date
            if period == 'monthly':
                start_date = datetime.now().replace(day=1).strftime('%Y-%m-%d')
            elif period == 'weekly':
                start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            else:  # yearly
                start_date = datetime.now().replace(month=1, day=1).strftime('%Y-%m-%d')
            
            # Get spent amount in this period
            cursor.execute('''
                SELECT COALESCE(SUM(amount), 0) FROM expenses
                WHERE user_id = ? AND category = ? AND date >= ?
            ''', (user_id, category, start_date))
            
            spent = cursor.fetchone()[0]
            remaining = budget_amount - spent
            percentage = (spent / budget_amount) * 100 if budget_amount > 0 else 0
            
            status_emoji = "🔴" if percentage >= 100 else "🟡" if percentage >= 80 else "🟢"
            
            response += f"{status_emoji} {category.title()} ({period}):\n"
            response += f"  Budget: ₹{budget_amount:.2f}\n"
            response += f"  Spent: ₹{spent:.2f} ({percentage:.1f}%)\n"
            response += f"  Remaining: ₹{remaining:.2f}\n\n"
        
        conn.close()
        
        return {
            'response': response,
            'intent': 'budget_check'
        }
    
    async def _handle_create_chart(self, message: str, user_id: str) -> Dict[str, Any]:
        """Create visualization charts"""
        
        conn = sqlite3.connect(self.db.db_path)
        
        # Get expense data for charts
        df = pd.read_sql_query('''
            SELECT category, amount, date, source
            FROM expenses
            WHERE user_id = ?
            AND date >= date('now', '-30 days')
            ORDER BY date
        ''', conn, params=(user_id,))
        
        conn.close()
        
        if df.empty:
            return {
                'response': "No expense data found for charts. Add some expenses first!",
                'intent': 'create_chart'
            }
        
        # Create category pie chart
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Category breakdown pie chart
            category_totals = df.groupby('category')['amount'].sum()
            colors = plt.cm.Set3(range(len(category_totals)))
            
            ax1.pie(category_totals.values, labels=category_totals.index, 
                   autopct='%1.1f%%', colors=colors, startangle=90)
            ax1.set_title('Expenses by Category (Last 30 Days)', fontsize=14, fontweight='bold')
            
            # Daily spending trend
            df['date'] = pd.to_datetime(df['date'])
            daily_spending = df.groupby('date')['amount'].sum().reset_index()
            
            ax2.plot(daily_spending['date'], daily_spending['amount'], 
                    marker='o', linewidth=2, markersize=4)
            ax2.set_title('Daily Spending Trend', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Amount (₹)')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save chart to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            total_expenses = df['amount'].sum()
            avg_daily = daily_spending['amount'].mean()
            
            return {
                'response': f"📊 Expense Analysis (Last 30 Days):\n"
                           f"• Total: ₹{total_expenses:.2f}\n"
                           f"• Daily Average: ₹{avg_daily:.2f}\n"
                           f"• Top Category: {category_totals.index[0]} (₹{category_totals.iloc[0]:.2f})",
                'intent': 'create_chart',
                'chart_data': img_base64,
                'stats': {
                    'total': total_expenses,
                    'daily_avg': avg_daily,
                    'top_category': category_totals.index[0]
                }
            }
            
        except Exception as e:
            logger.error(f"Chart creation failed: {e}")
            return {
                'response': "Sorry, couldn't create the chart. Please try again.",
                'intent': 'create_chart_failed'
            }
    
    async def _handle_general_query(self, message: str, user_id: str) -> Dict[str, Any]:
        """Handle general financial queries using AI"""
        
        # Get user's recent expense context
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT category, SUM(amount) as total, COUNT(*) as count
            FROM expenses
            WHERE user_id = ?
            AND date >= date('now', '-30 days')
            GROUP BY category
            ORDER BY total DESC
            LIMIT 5
        ''', (user_id,))
        
        expense_summary = cursor.fetchall()
        
        cursor.execute('''
            SELECT category, amount, period FROM budgets
            WHERE user_id = ?
        ''', (user_id,))
        
        budgets = cursor.fetchall()
        conn.close()
        
        # Create context for AI
        context = f"""
        User's recent expenses (last 30 days):
        {[f"{cat}: ₹{total:.2f} ({count} transactions)" for cat, total, count in expense_summary]}
        
        User's budgets:
        {[f"{cat}: ₹{amount:.2f} per {period}" for cat, amount, period in budgets]}
        """
        
        # Use AI for personalized response
        prompt = f"""
        You are a helpful financial assistant. Based on the user's financial context and their message, 
        provide helpful, personalized advice.
        
        User's financial context:
        {context}
        
        User message: "{message}"
        
        Provide a helpful, encouraging response with actionable financial advice.
        """
        
        try:
            ai_response = self.llm.predict(prompt)
            return {
                'response': ai_response,
                'intent': 'general_query'
            }
        except Exception as e:
            logger.error(f"AI query failed: {e}")
            return {
                'response': "I'm here to help with your finances! You can:\n"
                           "• Add expenses: 'I spent ₹500 on food'\n"
                           "• View expenses: 'Show my expenses'\n"
                           "• Set budgets: 'Set ₹5000 monthly budget for food'\n"
                           "• Check budgets: 'Check my budgets'\n"
                           "• Fetch SMS: 'Fetch my SMS transactions'",
                'intent': 'help'
            }
    
    async def register_device(self, device_data: DeviceRegistration) -> Dict[str, str]:
        """Register a new mobile device for SMS monitoring"""
        
        device_id = self.db.register_device({
            'user_id': device_data.user_id,
            'device_name': device_data.device_name,
            'device_type': device_data.device_type,
            'phone_number': device_data.phone_number
        })
        
        self.connected_devices[device_id] = {
            'user_id': device_data.user_id,
            'device_name': device_data.device_name,
            'status': 'connected',
            'connected_at': datetime.now()
        }
        
        return {
            'device_id': device_id,
            'status': 'registered',
            'message': f"Device '{device_data.device_name}' registered successfully!"
        }
    
    async def handle_sms_message(self, sms_data: SMSMessage) -> Dict[str, Any]:
        """Handle incoming SMS message from mobile device"""
        
        # Store SMS in database
        sms_id = self.db.store_sms_message({
            'device_id': sms_data.device_id,
            'user_id': self.connected_devices.get(sms_data.device_id, {}).get('user_id', 'unknown'),
            'sender': sms_data.sender,
            'message': sms_data.message,
            'timestamp': sms_data.timestamp
        })
        
        # Process if it's a banking SMS
        if sms_data.is_banking_sms or self.db._is_banking_sms(sms_data.message):
            result = self.sms_processor.process_sms_with_ai(
                sms_data.message, 
                sms_data.sender
            )
            
            if result:
                # Auto-create expense
                user_id = self.connected_devices.get(sms_data.device_id, {}).get('user_id')
                if user_id:
                    expense_id = self.db.add_expense({
                        'user_id': user_id,
                        'amount': result['amount'],
                        'category': result['category'],
                        'description': result['description'],
                        'merchant': result['merchant'],
                        'source': 'mobile_sms',
                        'confidence_score': result['confidence_score'],
                        'device_id': sms_data.device_id,
                        'sms_id': sms_id
                    })
                    
                    # Mark as processed
                    self.db.mark_sms_processed(sms_id, expense_id)
                    
                    # Send notification via WebSocket if connected
                    if user_id in self.websocket_connections:
                        notification = {
                            'type': 'new_expense',
                            'data': {
                                'amount': result['amount'],
                                'category': result['category'],
                                'merchant': result['merchant'],
                                'source': 'SMS'
                            }
                        }
                        await self.websocket_connections[user_id].send_text(
                            json.dumps(notification)
                        )
                    
                    return {
                        'status': 'processed',
                        'expense_created': True,
                        'expense_id': expense_id,
                        'amount': result['amount'],
                        'category': result['category']
                    }
        
        return {
            'status': 'stored',
            'sms_id': sms_id,
            'is_banking': sms_data.is_banking_sms
        }

# ====================================================================
# FASTAPI APPLICATION
# ====================================================================

app = FastAPI(title="Integrated Financial Assistant", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the assistant
# Note: In production, use environment variables for API keys
assistant = IntegratedFinancialAssistant(
    model_provider="gemini",  # or "cohere"
    api_key=os.getenv("GOOGLE_API_KEY") or "your-api-key-here"
)

# ====================================================================
# API ENDPOINTS
# ====================================================================

@app.post("/chat")
async def chat_endpoint(message_data: ChatMessage):
    """Main chat endpoint"""
    try:
        response = await assistant.process_chat_message(
            message_data.message,
            message_data.user_id,
            message_data.fetch_sms
        )
        return response
    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process message")

@app.post("/register-device")
async def register_device_endpoint(device_data: DeviceRegistration):
    """Register mobile device for SMS monitoring"""
    try:
        result = await assistant.register_device(device_data)
        return result
    except Exception as e:
        logger.error(f"Device registration failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to register device")

@app.post("/sms-webhook")
async def sms_webhook(sms_data: SMSMessage):
    """Webhook for receiving SMS messages from mobile devices"""
    try:
        result = await assistant.handle_sms_message(sms_data)
        return result
    except Exception as e:
        logger.error(f"SMS handling failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process SMS")

@app.post("/add-expense")
async def add_expense_endpoint(expense_data: ExpenseEntry):
    """Manually add expense"""
    try:
        expense_id = assistant.db.add_expense({
            'user_id': expense_data.user_id,
            'amount': expense_data.amount,
            'category': expense_data.category.value,
            'description': expense_data.description,
            'merchant': expense_data.merchant,
            'source': expense_data.source.value,
            'confidence_score': expense_data.confidence_score
        })
        
        return {
            'status': 'success',
            'expense_id': expense_id,
            'message': 'Expense added successfully'
        }
    except Exception as e:
        logger.error(f"Add expense failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to add expense")

@app.get("/expenses/{user_id}")
async def get_expenses(user_id: str, limit: int = 50):
    """Get user's expenses"""
    try:
        conn = sqlite3.connect(assistant.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, amount, category, description, merchant, date, source, confidence_score
            FROM expenses
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (user_id, limit))
        
        expenses = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': exp[0],
                'amount': exp[1],
                'category': exp[2],
                'description': exp[3],
                'merchant': exp[4],
                'date': exp[5],
                'source': exp[6],
                'confidence_score': exp[7]
            }
            for exp in expenses
        ]
    except Exception as e:
        logger.error(f"Get expenses failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve expenses")

@app.post("/set-budget")
async def set_budget_endpoint(budget_data: BudgetSet):
    """Set budget for category"""
    try:
        conn = sqlite3.connect(assistant.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO budgets (user_id, category, amount, period)
            VALUES (?, ?, ?, ?)
        ''', (budget_data.user_id, budget_data.category, 
              budget_data.amount, budget_data.period))
        
        conn.commit()
        conn.close()
        
        return {
            'status': 'success',
            'message': f'Budget set: ₹{budget_data.amount} for {budget_data.category}'
        }
    except Exception as e:
        logger.error(f"Set budget failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to set budget")

@app.get("/budgets/{user_id}")
async def get_budgets(user_id: str):
    """Get user's budgets with status"""
    try:
        conn = sqlite3.connect(assistant.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT category, amount, period FROM budgets
            WHERE user_id = ?
        ''', (user_id,))
        
        budgets = cursor.fetchall()
        budget_status = []
        
        for category, amount, period in budgets:
            # Calculate period start
            if period == 'monthly':
                start_date = datetime.now().replace(day=1).strftime('%Y-%m-%d')
            elif period == 'weekly':
                start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            else:
                start_date = datetime.now().replace(month=1, day=1).strftime('%Y-%m-%d')
            
            # Get spent amount
            cursor.execute('''
                SELECT COALESCE(SUM(amount), 0) FROM expenses
                WHERE user_id = ? AND category = ? AND date >= ?
            ''', (user_id, category, start_date))
            
            spent = cursor.fetchone()[0]
            
            budget_status.append({
                'category': category,
                'budget': amount,
                'spent': spent,
                'remaining': amount - spent,
                'percentage': (spent / amount) * 100 if amount > 0 else 0,
                'period': period
            })
        
        conn.close()
        return budget_status
        
    except Exception as e:
        logger.error(f"Get budgets failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve budgets")

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket for real-time notifications"""
    await websocket.accept()
    assistant.websocket_connections[user_id] = websocket
    
    try:
        while True:
            # Keep connection alive and handle any messages
            data = await websocket.receive_text()
            
            # Echo back for testing
            await websocket.send_text(f"Echo: {data}")
            
    except WebSocketDisconnect:
        del assistant.websocket_connections[user_id]
        logger.info(f"WebSocket disconnected for user: {user_id}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Integrated Financial Assistant API",
        "status": "running",
        "version": "1.0.0",
        "features": [
            "AI-powered expense categorization",
            "SMS auto-fetch and processing",
            "Budget management",
            "Real-time notifications",
            "Expense visualization"
        ]
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected",
        "ai_model": "available",
        "connected_devices": len(assistant.connected_devices),
        "active_websockets": len(assistant.websocket_connections)
    }

# ====================================================================
# BACKGROUND TASKS
# ====================================================================

async def process_pending_sms():
    """Background task to process any pending SMS messages"""
    while True:
        try:
            # Process for all users with SMS permissions
            conn = sqlite3.connect(assistant.db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT DISTINCT user_id FROM users 
                WHERE allow_sms_fetch = 1
            ''')
            
            users = cursor.fetchall()
            conn.close()
            
            for (user_id,) in users:
                await assistant.fetch_and_process_sms(user_id)
            
            # Wait 5 minutes before next check
            await asyncio.sleep(300)
            
        except Exception as e:
            logger.error(f"Background SMS processing failed: {e}")
            await asyncio.sleep(60)  # Wait 1 minute on error

# ====================================================================
# STARTUP EVENT
# ====================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize background tasks on startup"""
    logger.info("Starting Integrated Financial Assistant...")
    
    # Start background task for SMS processing
    asyncio.create_task(process_pending_sms())
    
    logger.info("Financial Assistant started successfully!")

# ====================================================================
# MAIN EXECUTION
# ====================================================================

if __name__ == "__main__":
    print("🤖 Integrated Financial Assistant with SMS Auto-Fetch")
    print("=" * 60)
    print("Features:")
    print("• AI-powered expense categorization")
    print("• Real-time SMS processing")
    print("• Budget management and alerts")
    print("• Expense visualization")
    print("• WebSocket notifications")
    print("=" * 60)
    
    uvicorn.run(
        "main:app",  # Assuming this file is named main.py
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
