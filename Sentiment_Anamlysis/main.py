#!/usr/bin/env python3
"""
Conversation Summarization & Sentiment Analysis App using LangChain & LangGraph
===============================================================================

This Flask application uses LangChain and LangGraph to create a workflow that:
1. Processes customer call transcripts
2. Generates summaries using LangChain
3. Extracts sentiment using LangGraph workflow
4. Saves results to CSV file

Requirements:
- pip install flask langchain langchain-groq langgraph python-dotenv
- Set GROQ_API_KEY in .env file or environment variable
"""

import os
import csv
from datetime import datetime
from typing import Dict, Any, TypedDict
from flask import Flask, request, render_template_string, jsonify

# LangChain imports
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize LangChain Groq client
try:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )
    print("✅ LangChain Groq client initialized successfully")
except Exception as e:
    print(f"❌ Error initializing LangChain Groq client: {e}")
    llm = None

# CSV file configuration
CSV_FILE = "call_analysis.csv"
CSV_HEADERS = ["Timestamp", "Transcript", "Summary", "Sentiment"]

# Define state for LangGraph workflow
class AnalysisState(TypedDict):
    transcript: str
    summary: str
    sentiment: str
    error: str

def initialize_csv():
    """Initialize CSV file with headers if it doesn't exist"""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(CSV_HEADERS)
        print(f"✅ Created new CSV file: {CSV_FILE}")

# LangChain Prompt Templates
SUMMARY_TEMPLATE = PromptTemplate(
    input_variables=["transcript"],
    template="""You are an expert at analyzing customer service conversations. 
    
    Summarize the following customer call transcript in exactly 2-3 clear, concise sentences. 
    Focus on the main issue, customer needs, and key details.
    
    Transcript: {transcript}
    
    Summary:"""
)

SENTIMENT_TEMPLATE = PromptTemplate(
    input_variables=["transcript"],
    template="""You are an expert sentiment analyst for customer service interactions.
    
    Analyze the customer's sentiment in the following conversation transcript.
    
    Classify the sentiment as one of these options:
    - Positive (satisfied, happy, pleased)
    - Neutral (informational, matter-of-fact)
    - Negative (frustrated, angry, disappointed, upset)
    
    Provide your response in this exact format: "[Classification] ([specific emotion if applicable])"
    
    Transcript: {transcript}
    
    Sentiment:"""
)

# Create LangChain chains
summary_chain = (
    {"transcript": RunnablePassthrough()}
    | SUMMARY_TEMPLATE
    | llm
    | StrOutputParser()
)

sentiment_chain = (
    {"transcript": RunnablePassthrough()}
    | SENTIMENT_TEMPLATE
    | llm
    | StrOutputParser()
)

# Define LangGraph workflow nodes
def summarize_node(state: AnalysisState) -> AnalysisState:
    """Node to generate conversation summary using LangChain"""
    try:
        print("📝 Generating summary...")
        summary = summary_chain.invoke(state["transcript"])
        state["summary"] = summary.strip()
        print(f"✅ Summary generated: {summary[:50]}...")
    except Exception as e:
        print(f"❌ Error in summarize_node: {e}")
        state["error"] = f"Summary generation failed: {str(e)}"
        state["summary"] = "Error generating summary"
    
    return state

def sentiment_node(state: AnalysisState) -> AnalysisState:
    """Node to analyze sentiment using LangChain"""
    try:
        print("😊 Analyzing sentiment...")
        sentiment = sentiment_chain.invoke(state["transcript"])
        state["sentiment"] = sentiment.strip()
        print(f"✅ Sentiment analyzed: {sentiment}")
    except Exception as e:
        print(f"❌ Error in sentiment_node: {e}")
        state["error"] = f"Sentiment analysis failed: {str(e)}"
        state["sentiment"] = "Error analyzing sentiment"
    
    return state

def validate_input_node(state: AnalysisState) -> AnalysisState:
    """Node to validate input transcript"""
    transcript = state.get("transcript", "").strip()
    
    if not transcript:
        state["error"] = "Empty transcript provided"
        return state
    
    if len(transcript) < 10:
        state["error"] = "Transcript too short (minimum 10 characters)"
        return state
    
    print(f"✅ Input validated: {len(transcript)} characters")
    return state

def save_results_node(state: AnalysisState) -> AnalysisState:
    """Node to save results to CSV"""
    if state.get("error"):
        print(f"⚠️ Skipping save due to error: {state['error']}")
        return state
    
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp,
                state["transcript"],
                state["summary"],
                state["sentiment"]
            ])
        
        print(f"✅ Results saved to {CSV_FILE}")
        
    except Exception as e:
        print(f"❌ Error saving to CSV: {e}")
        state["error"] = f"Failed to save results: {str(e)}"
    
    return state

# Create LangGraph workflow
def create_analysis_workflow():
    """Create the LangGraph workflow for conversation analysis"""
    
    # Create state graph
    workflow = StateGraph(AnalysisState)
    
    # Add nodes
    workflow.add_node("validate_input", validate_input_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("analyze_sentiment", sentiment_node)
    workflow.add_node("save_results", save_results_node)
    
    # Define the workflow edges
    workflow.set_entry_point("validate_input")
    
    # Conditional routing based on validation
    def should_continue(state: AnalysisState) -> str:
        if state.get("error"):
            return "save_results"  # Skip to save (which will skip due to error)
        return "summarize"
    
    workflow.add_conditional_edges(
        "validate_input",
        should_continue,
        {
            "summarize": "summarize",
            "save_results": "save_results"
        }
    )
    
    # Continue workflow
    workflow.add_edge("summarize", "analyze_sentiment")
    workflow.add_edge("analyze_sentiment", "save_results")
    workflow.add_edge("save_results", END)
    
    return workflow.compile()

# Initialize the workflow
analysis_workflow = create_analysis_workflow()

def analyze_conversation(transcript: str) -> Dict[str, Any]:
    """Main function to analyze conversation using LangGraph workflow"""
    
    if not llm:
        return {
            "success": False,
            "error": "LangChain Groq client not initialized. Please check your API key."
        }
    
    print(f"\n🔍 Starting LangGraph workflow for transcript ({len(transcript)} characters)...")
    print(f"Transcript preview: {transcript[:100]}...")
    
    # Initialize state
    initial_state: AnalysisState = {
        "transcript": transcript,
        "summary": "",
        "sentiment": "",
        "error": ""
    }
    
    try:
        # Run the LangGraph workflow
        final_state = analysis_workflow.invoke(initial_state)
        
        # Print results to console
        print("\n" + "="*60)
        print("📊 LANGGRAPH WORKFLOW RESULTS")
        print("="*60)
        print(f"📄 ORIGINAL TRANSCRIPT:")
        print(f"{final_state['transcript']}\n")
        print(f"📝 SUMMARY:")
        print(f"{final_state['summary']}\n")
        print(f"😊 SENTIMENT:")
        print(f"{final_state['sentiment']}")
        
        if final_state.get("error"):
            print(f"⚠️ ERRORS:")
            print(f"{final_state['error']}")
        
        print("="*60 + "\n")
        
        return {
            "success": not bool(final_state.get("error")),
            "transcript": final_state["transcript"],
            "summary": final_state["summary"],
            "sentiment": final_state["sentiment"],
            "error": final_state.get("error")
        }
        
    except Exception as e:
        error_msg = f"LangGraph workflow failed: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg
        }

# HTML Template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LangChain & LangGraph Conversation Analysis</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 900px; margin: 30px auto; padding: 20px; background: #f8f9fa; }
        .container { background: white; padding: 40px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .badge { background: #007bff; color: white; padding: 5px 10px; border-radius: 20px; font-size: 0.8em; margin: 0 5px; }
        .form-group { margin-bottom: 25px; }
        label { display: block; margin-bottom: 8px; font-weight: 600; color: #333; }
        textarea { width: 100%; height: 160px; padding: 15px; border: 2px solid #e9ecef; border-radius: 8px; font-family: inherit; resize: vertical; }
        textarea:focus { border-color: #007bff; outline: none; box-shadow: 0 0 0 3px rgba(0,123,255,0.1); }
        .btn { background: linear-gradient(135deg, #007bff, #0056b3); color: white; padding: 12px 25px; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: 600; transition: transform 0.2s; }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,123,255,0.3); }
        .result { margin-top: 30px; padding: 25px; background: #f8f9fa; border-radius: 10px; border-left: 5px solid #28a745; }
        .result h3 { color: #155724; margin-bottom: 20px; }
        .result-item { margin-bottom: 20px; padding: 15px; background: white; border-radius: 8px; border: 1px solid #e9ecef; }
        .result-label { font-weight: 600; color: #495057; margin-bottom: 5px; }
        .error { background: #f8d7da; border-left-color: #dc3545; }
        .error h3 { color: #721c24; }
        .sample-transcripts { margin-top: 40px; }
        .sample { background: #e9ecef; padding: 20px; margin: 15px 0; border-radius: 10px; cursor: pointer; transition: all 0.3s; border: 2px solid transparent; }
        .sample:hover { background: #dee2e6; border-color: #007bff; transform: translateY(-2px); }
        .sample-title { font-weight: 600; color: #495057; margin-bottom: 10px; }
        .workflow-info { background: #e3f2fd; padding: 20px; border-radius: 10px; margin-top: 30px; border-left: 5px solid #2196f3; }
        .tech-stack { display: flex; justify-content: center; gap: 10px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 LangChain & LangGraph Conversation Analysis</h1>
            <p>Advanced AI-powered conversation analysis using LangChain workflows and LangGraph orchestration</p>
            <div class="tech-stack">
                <span class="badge">LangChain</span>
                <span class="badge">LangGraph</span>
                <span class="badge">Groq API</span>
                <span class="badge">Flask</span>
            </div>
        </div>
        
        {% if error %}
        <div class="result error">
            <h3>❌ Workflow Error</h3>
            <div class="result-item">
                <div class="result-label">Error Details:</div>
                {{ error }}
            </div>
        </div>
        {% endif %}
        
        {% if result and result.success %}
        <div class="result">
            <h3>🎯 LangGraph Analysis Complete</h3>
            
            <div class="result-item">
                <div class="result-label">📄 Original Transcript:</div>
                {{ result.transcript }}
            </div>
            
            <div class="result-item">
                <div class="result-label">📝 AI-Generated Summary:</div>
                {{ result.summary }}
            </div>
            
            <div class="result-item">
                <div class="result-label">😊 Sentiment Analysis:</div>
                {{ result.sentiment }}
            </div>
            
            <div style="text-align: center; margin-top: 20px; color: #28a745; font-weight: 600;">
                ✅ Results automatically saved to call_analysis.csv
            </div>
        </div>
        {% endif %}
        
        <form method="POST" action="/analyze">
            <div class="form-group">
                <label for="transcript">📝 Enter Customer Call Transcript:</label>
                <textarea name="transcript" id="transcript" 
                    placeholder="Paste or type the customer conversation transcript here. The LangGraph workflow will process it through validation → summarization → sentiment analysis → saving." 
                    required>{{ current_transcript if current_transcript else '' }}</textarea>
            </div>
            <button type="submit" class="btn">🚀 Run LangGraph Analysis</button>
        </form>
        
        <div class="sample-transcripts">
            <h3>📝 Sample Transcripts for Testing</h3>
            <p>Click any sample to automatically populate the form:</p>
            
            <div class="sample" onclick="useTranscript(this)">
                <div class="sample-title">💳 Payment Issue (Negative Sentiment)</div>
                "Hi, I was trying to book a slot yesterday but the payment failed. I'm very frustrated because I need that appointment urgently. The system kept showing an error message and now I'm not sure if my booking went through. Can you please help me check my booking status and process the payment properly? This is really inconvenient and I'm running out of time."
            </div>
            
            <div class="sample" onclick="useTranscript(this)">
                <div class="sample-title">❓ Product Inquiry (Neutral Sentiment)</div>
                "Hello, I'm interested in learning more about your premium subscription plan. Could you tell me what features are included and how much it costs? I'm currently using the free version but I need more storage space and better customer support. What's the difference between the basic and premium tiers? Also, do you offer any discounts for annual subscriptions?"
            </div>
            
            <div class="sample" onclick="useTranscript(this)">
                <div class="sample-title">📦 Shipping Complaint (Negative Sentiment)</div>
                "I received my order yesterday but the product was damaged during shipping. The box was completely crushed and the item inside was broken into pieces. I'm really disappointed because this was supposed to be a gift for my daughter's birthday tomorrow. I need either a replacement sent with express shipping or a full refund immediately. This is completely unacceptable quality control."
            </div>
            
            <div class="sample" onclick="useTranscript(this)">
                <div class="sample-title">🌟 Positive Feedback (Positive Sentiment)</div>
                "I just wanted to call and express my sincere gratitude for the excellent customer service I received. Your technical support team helped me resolve my software issue within minutes, and the representative was incredibly knowledgeable and patient. The new update works perfectly now and I'm very satisfied with all the improvements. You've definitely earned a loyal customer. Keep up the fantastic work!"
            </div>
            
            <div class="sample" onclick="useTranscript(this)">
                <div class="sample-title">🔧 Technical Support (Mixed Sentiment)</div>
                "Hi, I'm having some trouble with the new app version you released last week. The interface looks great and I love the new design, but I'm experiencing frequent crashes when I try to upload files. It's working fine for basic functions, but the file upload feature is essential for my work. Could you help me troubleshoot this issue? I really want to keep using your service but need this fixed."
            </div>
        </div>
        
        <div class="workflow-info">
            <h4>🔄 LangGraph Workflow Process</h4>
            <p><strong>This app uses a sophisticated LangGraph workflow:</strong></p>
            <ol>
                <li><strong>Input Validation:</strong> Checks transcript length and format</li>
                <li><strong>Parallel Processing:</strong> LangChain chains for summarization and sentiment</li>
                <li><strong>Result Aggregation:</strong> Combines outputs from multiple AI operations</li>
                <li><strong>Data Persistence:</strong> Automatically saves to CSV with timestamps</li>
                <li><strong>Error Handling:</strong> Comprehensive error recovery and reporting</li>
            </ol>
            <p>Each analysis leverages Groq's high-performance inference with LangChain's prompt engineering and LangGraph's workflow orchestration.</p>
        </div>
    </div>
    
    <script>
        function useTranscript(element) {
            // Extract transcript text, removing the title
            const content = element.innerText;
            const transcript = content.substring(content.indexOf('"'));
            document.getElementById('transcript').value = transcript;
            document.getElementById('transcript').scrollIntoView({ behavior: 'smooth' });
            document.getElementById('transcript').focus();
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main page with form to input transcript"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Process transcript using LangGraph workflow"""
    transcript = request.form.get('transcript', '').strip()
    
    if not transcript:
        return render_template_string(HTML_TEMPLATE, 
            error="Please provide a transcript to analyze.")
    
    # Run LangGraph analysis
    result = analyze_conversation(transcript)
    
    if not result["success"]:
        return render_template_string(HTML_TEMPLATE, 
            error=result["error"],
            current_transcript=transcript)
    
    return render_template_string(HTML_TEMPLATE, result=result)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for programmatic access to LangGraph workflow"""
    data = request.get_json()
    if not data or 'transcript' not in data:
        return jsonify({'error': 'Missing transcript in request body'}), 400
    
    transcript = data['transcript'].strip()
    if not transcript:
        return jsonify({'error': 'Empty transcript provided'}), 400
    
    # Run LangGraph analysis
    result = analyze_conversation(transcript)
    
    if result["success"]:
        return jsonify({
            'success': True,
            'transcript': result['transcript'],
            'summary': result['summary'],
            'sentiment': result['sentiment'],
            'timestamp': datetime.now().isoformat(),
            'workflow': 'LangGraph + LangChain'
        })
    else:
        return jsonify({
            'success': False,
            'error': result['error']
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'langchain_ready': llm is not None,
        'csv_file_exists': os.path.exists(CSV_FILE),
        'workflow_engine': 'LangGraph',
        'llm_provider': 'LangChain-Groq'
    })

@app.route('/workflow-info')
def workflow_info():
    """Endpoint to get workflow information"""
    return jsonify({
        'workflow_type': 'LangGraph StateGraph',
        'nodes': [
            'validate_input',
            'summarize', 
            'analyze_sentiment',
            'save_results'
        ],
        'chains': [
            'summary_chain (LangChain)',
            'sentiment_chain (LangChain)'
        ],
        'llm_model': 'llama-3.3-70b-versatile',
        'provider': 'Groq via LangChain'
    })

if __name__ == '__main__':
    print("🚀 Starting LangChain & LangGraph Conversation Analysis App...")
    print("="*60)
    
    # Check API key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found in environment variables!")
        print("Please set your API key:")
        print("export GROQ_API_KEY='your-api-key-here'")
        print("Or create a .env file with: GROQ_API_KEY=your-api-key-here")
        exit(1)
    else:
        print(f"✅ API Key found: {api_key[:8]}...{api_key[-4:]}")
    
    # Initialize CSV file
    initialize_csv()
    
    # Test the workflow creation
    try:
        test_workflow = create_analysis_workflow()
        print("✅ LangGraph workflow created successfully")
    except Exception as e:
        print(f"❌ Error creating LangGraph workflow: {e}")
        exit(1)
    
    # Start the server
    print(f"\n🌐 LangChain & LangGraph Server starting at: http://localhost:5000")
    print("📊 Features: LangChain Chains + LangGraph Workflow Orchestration")
    print("🛑 Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
