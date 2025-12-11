
class Config:
    # Gemini API settings
    GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
    GEMINI_MODEL = "gemini-2.5-flash"
    GEMINI_COST_PER_1K_TOKENS = 0.00015  # Approximate cost
    
    # Scoring weights
    SCORE_WEIGHTS = {
        "relevance": 0.4,
        "hallucination": 0.4,
        "performance": 0.2
    }
    
    # Performance thresholds
    MAX_LATENCY_MS = 3000
    MAX_COST_PER_RESPONSE = 0.01