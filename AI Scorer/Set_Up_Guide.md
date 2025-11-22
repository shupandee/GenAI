# Complete Setup Guide - Communication Skills Scoring System

## 📦 What You're Building

A backend AI system that:
1. Accepts student self-introduction transcripts
2. Analyzes them using NLP and rule-based methods
3. Scores them on 5 criteria (0-100 scale)
4. Provides detailed feedback

## 🎯 Quick Start (5 Minutes)

```bash
# 1. Download/clone the project
git clone <repo-url>
cd communication-scorer

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies (takes 5-10 min)
pip install -r requirements.txt

# 5. Test the system
python test_scorer.py

# 6. Run the API
python app.py
```

That's it! API is now running at http://localhost:5000

## 📁 Project Files Explained

```
communication-scorer/
│
├── app.py                    # Flask API server (entry point)
├── scorer.py                 # Core AI scoring logic
├── requirements.txt          # Python dependencies
├── test_scorer.py           # Test script
├── api_client.py            # Sample API usage
├── Dockerfile               # Docker configuration
│
├── README.md                # Project documentation
├── DEPLOYMENT.md            # Deployment instructions
└── SETUP_GUIDE.md           # This file
```

## 🔧 Detailed Setup Instructions

### Prerequisites

Before starting, ensure you have:

✅ **Python 3.8 or higher**
```bash
python --version
# or
python3 --version
```

✅ **pip (Python package manager)**
```bash
pip --version
```

✅ **At least 2GB free RAM**
- NLP models require memory to load

✅ **Stable internet connection**
- For downloading models (first time only)

### Step 1: Get the Code

#### Option A: Using Git
```bash
git clone <your-repository-url>
cd communication-scorer
```

#### Option B: Download ZIP
1. Download the project ZIP file
2. Extract to a folder
3. Open terminal/command prompt in that folder

### Step 2: Create Virtual Environment

**Why?** Keeps project dependencies isolated from your system Python.

#### Windows
```cmd
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Troubleshooting Virtual Environment

**Issue**: "venv command not found"
```bash
# Install venv
sudo apt-get install python3-venv  # Ubuntu/Debian
```

**Issue**: Activation script not found
```bash
# Recreate the virtual environment
rm -rf venv
python -m venv venv
```

### Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

**What gets installed:**
- Flask (web framework): ~10MB
- sentence-transformers (NLP): ~80MB
- LanguageTool (grammar): ~200MB
- PyTorch (ML framework): ~500MB
- Other dependencies: ~50MB

**Total download**: ~800MB (one-time)
**Time**: 5-15 minutes depending on internet speed

#### Troubleshooting Installation

**Issue**: "No module named 'pip'"
```bash
python -m ensurepip --upgrade
```

**Issue**: Timeout during download
```bash
# Increase timeout
pip install --timeout=120 -r requirements.txt
```

**Issue**: "Insufficient disk space"
- Free up at least 2GB of disk space
- Or use lighter models (modify scorer.py)

**Issue**: Build errors on Windows
```bash
# Install Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### Step 4: Verify Installation

```bash
python test_scorer.py
```

**Expected output:**
```
Communication Skills Scorer - Test Script
================================================================================

Initializing scorer...
Loading sentence transformer model...
Loading grammar checker...
Loading sentiment analyzer...
✓ Scorer initialized successfully!
...
Overall Score: 86.0/100
...
TEST COMPLETED SUCCESSFULLY!
```

**If you see errors:**
1. Check error message carefully
2. Verify all dependencies installed: `pip list`
3. Try reinstalling: `pip install --force-reinstall -r requirements.txt`

### Step 5: Run the API Server

```bash
python app.py
```

**Expected output:**
```
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server.
 * Running on http://0.0.0.0:5000
Press CTRL+C to quit
```

**Server is ready!** Keep this terminal window open.

### Step 6: Test the API

Open a **new terminal window** (keep server running in first one):

#### Test 1: Health Check
```bash
curl http://localhost:5000/api/health
```

Expected response:
```json
{"status": "healthy", "scorer_ready": true}
```

#### Test 2: Score a Transcript

**Windows PowerShell:**
```powershell
$body = @{transcript="Hello everyone, myself Muskan, studying in class 8th..."} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:5000/api/score" -Method Post -Body $body -ContentType "application/json"
```

**macOS/Linux:**
```bash
curl -X POST http://localhost:5000/api/score \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Hello everyone, myself Muskan, studying in class 8th B section from Christ Public School. I am 13 years old..."}'
```

#### Test 3: Using Python Client
```bash
# In new terminal (with venv activated)
python api_client.py
```

## 🎮 Usage Examples

### Example 1: Basic API Call (Python)

```python
import requests

transcript = """Hello everyone, myself Muskan, studying in class 8th B section from Christ Public School. 
I am 13 years old. I live with my family..."""

response = requests.post(
    'http://localhost:5000/api/score',
    json={'transcript': transcript}
)

result = response.json()
print(f"Overall Score: {result['overall_score']}")
```

### Example 2: Command Line (cURL)

```bash
curl -X POST http://localhost:5000/api/score \
  -H "Content-Type: application/json" \
  -d @transcript.txt
```

Where `transcript.txt` contains:
```json
{"transcript": "Your transcript text here..."}
```

### Example 3: JavaScript (Node.js)

```javascript
const axios = require('axios');

const transcript = "Hello everyone, myself Muskan...";

axios.post('http://localhost:5000/api/score', {
  transcript: transcript
})
.then(response => {
  console.log('Overall Score:', response.data.overall_score);
})
.catch(error => {
  console.error('Error:', error);
});
```

## 🐛 Common Issues & Solutions

### Issue 1: Port 5000 Already in Use

**Error:** `Address already in use`

**Solution:**
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:5000 | xargs kill -9

# Or change port in app.py
# Change: port = int(os.environ.get('PORT', 5000))
# To:     port = int(os.environ.get('PORT', 8000))
```

### Issue 2: Models Not Downloading

**Error:** "Connection timeout" or "Model not found"

**Solution:**
1. Check internet connection
2. Try manual download:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

### Issue 3: Out of Memory

**Error:** "Killed" or "MemoryError"

**Solutions:**
- Close other applications
- Use lighter model (edit scorer.py line 13):
  ```python
  self.semantic_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
  ```
- Increase system swap space (Linux)

### Issue 4: Slow First Request

**Cause:** Models loading into memory

**Normal:** First request takes 5-10 seconds
**Solution:** Preload models (see scorer.py)

### Issue 5: CORS Error (Browser)

**Error:** "Cross-Origin Request Blocked"

**Solution:** Already handled with Flask-CORS, but if issues persist:
```python
# In app.py
CORS(app, resources={r"/api/*": {"origins": "*"}})
```

## 📊 Understanding the Output

### Sample Response Structure

```json
{
  "overall_score": 86.0,
  "words": 120,
  "sentences": 11,
  "duration_seconds": 52,
  "criteria": [
    {
      "criterion": "Content & Structure",
      "score": 28.5,
      "max_score": 30,
      "weighted_score": 38.0,
      "weight": 40,
      "details": {
        "salutation": {...},
        "keywords": {...},
        "flow": {...}
      }
    },
    ...
  ]
}
```

### Score Interpretation

- **90-100**: Excellent communication skills
- **80-89**: Very good, minor improvements needed
- **70-79**: Good, some areas need work
- **60-69**: Average, significant improvements needed
- **<60**: Needs substantial improvement

## 🔄 Next Steps

### 1. Modify Rubric
Edit `scorer.py` to customize scoring criteria:
- Line 40-60: Rubric descriptions
- Line 100-150: Content scoring
- Line 200-250: Other criteria

### 2. Deploy
See `DEPLOYMENT.md` for cloud deployment options:
- AWS EC2
- Heroku
- Railway
- Docker

### 3. Add Features
Ideas for enhancement:
- Authentication (API keys)
- Rate limiting
- Database storage
- Batch processing
- Real-time audio transcription

### 4. Integration
Integrate with:
- Frontend applications
- Mobile apps
- Learning management systems
- Educational platforms

## 📚 Additional Resources

### Documentation
- [Flask Documentation](https://flask.palletsprojects.com/)
- [sentence-transformers](https://www.sbert.net/)
- [LanguageTool](https://languagetool.org/)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)

### Tutorials
- REST API basics
- NLP fundamentals
- Flask deployment

## 💡 Tips for Success

1. **Keep virtual environment activated** while working
2. **Test locally** before deploying
3. **Monitor logs** for errors
4. **Version control** with Git
5. **Document changes** you make
6. **Backup regularly**

## 🆘 Getting Help

If you're stuck:

1. **Check error messages** carefully
2. **Review logs**: Look at terminal output
3. **Test components** individually (test_scorer.py)
4. **Verify dependencies**: `pip list`
5. **Check Python version**: `python --version`
6. **Google the error** with "Flask" or "Python"

## ✅ Final Checklist

Before submission, verify:

- [ ] All files present in repository
- [ ] README.md is complete
- [ ] Code runs without errors locally
- [ ] test_scorer.py passes
- [ ] API responds to health check
- [ ] Sample transcript scores correctly
- [ ] requirements.txt is updated
- [ ] Documentation is clear
- [ ] Screenshots/video recorded
- [ ] GitHub repository is public

## 🎉 You're All Set!

Your backend AI system is now running. The API is ready to accept transcripts and return detailed scores.

**Remember:** This is a case study to demonstrate your thinking and approach, not a test of perfect code. Show your problem-solving process!

Good luck! 🚀
