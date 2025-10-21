# config.py - Updated to use Gemini instead of OpenAI
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password_here"

# Using Gemini instead of OpenAI
GEMINI_API_KEY = "Api_key_here"
GEMINI_CHAT_MODEL = "gemini-2.5-flash"  # ADD THIS LINE

PINECONE_API_KEY = "API_key_here"
PINECONE_ENV = "us-east1-gcp"
PINECONE_INDEX_NAME = "vietnam-travel"
PINECONE_VECTOR_DIM = 768
EMBEDDING_DIMENSION = 768  

# system prompt for the chat model
SYSTEM_PROMPT = """You are an expert Vietnam travel assistant. Provide detailed, 
personalized travel recommendations based on the context provided."""