# 🚀 Complete Setup Guide - Hybrid AI Travel Assistant

## Prerequisites

- Python 3.8 or higher
- Neo4j Database (Community or Enterprise)
- Gemini API Key (free tier available)
- Pinecone API Key (free tier available)

---

## Step 1: Environment Setup

### 1.1 Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 1.2 Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt contents:**
```
neo4j==5.9.0
google-generativeai==0.3.2
pinecone-client==3.0.0
pyvis==0.3.1
networkx==3.1
tqdm
python-dotenv
aiohttp==3.9.1
```

---

## Step 2: Neo4j Setup

### 2.1 Install Neo4j

**Option A: Neo4j Desktop (Recommended for beginners)**
1. Download from: https://neo4j.com/download/
2. Install and launch Neo4j Desktop
3. Create a new project
4. Create a new database (set password)
5. Start the database

**Option B: Docker**
```bash
docker run \
    --name neo4j-travel \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    -v $HOME/neo4j/data:/data \
    neo4j:latest
```

### 2.2 Verify Connection

Open browser: http://localhost:7474
- Username: `neo4j`
- Password: `password` (or your custom password)

---

## Step 3: API Keys Configuration

### 3.1 Get Gemini API Key

1. Go to: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy your API key

### 3.2 Get Pinecone API Key

1. Sign up at: https://www.pinecone.io/
2. Go to API Keys section
3. Copy your API key
4. Note your environment (e.g., `us-east-1`)

### 3.3 Configure config.py

Create or edit `config.py`:

```python
# config.py
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your-neo4j-password"  # Change this!

GEMINI_API_KEY = "your-gemini-api-key-here"  # Paste your key
PINECONE_API_KEY = "your-pinecone-api-key"   # Paste your key
PINECONE_ENV = "us-east-1"                    # Your Pinecone environment
PINECONE_INDEX_NAME = "vietnam-travel"
PINECONE_VECTOR_DIM = 768
```

**⚠️ Security Note**: Never commit `config.py` with real keys to version control!

---

## Step 4: Load Data

### 4.1 Verify Data File

Ensure `vietnam_travel_dataset.json` is in your project directory.

### 4.2 Load to Neo4j

```bash
python load_to_neo4j.py
```

**Expected Output:**
```
Creating nodes: 100%|████████████| 350/350 [00:15<00:00]
Creating relationships: 100%|████████████| 350/350 [00:20<00:00]
Done loading into Neo4j.
```

### 4.3 Visualize Graph (Optional)

```bash
python visualize_graph.py
```

Opens `neo4j_viz.html` in your browser showing the graph structure.

### 4.4 Upload to Pinecone

```bash
python pinecone_upload.py
```

**Expected Output:**
```
Creating serverless index: vietnam-travel
Waiting for index to be ready...
Preparing to upsert 350 items to Pinecone...
Uploading batches: 100%|████████████| 11/11 [00:45<00:00]
All items uploaded successfully.
Index stats: {'dimension': 768, 'index_fullness': 0.0, 'namespaces': {'': {'vector_count': 350}}}
```

---

## Step 5: Run the Chat Assistant

```bash
python hybrid_chat.py
```

**Expected Output:**
```
============================================================
🇻🇳  HYBRID VIETNAM TRAVEL ASSISTANT 🇻🇳
============================================================

Powered by: Gemini AI + Pinecone + Neo4j
Type 'exit' or 'quit' to end the session
Type 'help' for example queries

============================================================

🗺️  Enter your travel question:
```

---

## Step 6: Test the System

### Test Query 1: Simple
```
create a romantic 4 day itinerary for Vietnam
```

### Test Query 2: Specific
```
best hotels in Hoi An for couples
```

### Test Query 3: Activity
```
beach activities in Nha Trang
```

---

## Troubleshooting

### Issue: "Neo4j connection failed"

**Solution:**
1. Verify Neo4j is running: `neo4j status`
2. Check credentials in config.py
3. Test connection at http://localhost:7474

### Issue: "Pinecone index not found"

**Solution:**
```bash
# Delete old index if dimension is wrong
python
>>> from pinecone import Pinecone
>>> pc = Pinecone(api_key="your-key")
>>> pc.delete_index("vietnam-travel")
>>> exit()

# Re-run upload
python pinecone_upload.py
```

### Issue: "Gemini API error"

**Solution:**
1. Verify API key at https://makersuite.google.com/
2. Check quota limits
3. Ensure `google-generativeai` is installed

### Issue: "Module not found"

**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

### Issue: "Dimension mismatch"

**Solution:**
- Gemini uses 768 dimensions
- Ensure config.py has: `PINECONE_VECTOR_DIM = 768`
- Delete and recreate Pinecone index

---

## Performance Tips

### Faster Responses
1. Use SSD for Neo4j data directory
2. Increase Neo4j memory in neo4j.conf
3. Warm up cache by running test queries

### Lower Costs
1. Use embedding cache (already implemented)
2. Batch requests when possible
3. Use Gemini Flash (faster, cheaper than Pro)

---

## Next Steps

1. ✅ System is running
2. Try example queries
3. Review improvements.md for features
4. Customize prompts in hybrid_chat.py
5. Add your own travel data

---

## Quick Start Script

Save as `quick_start.sh`:

```bash
#!/bin/bash

echo "🚀 Starting Hybrid AI Travel Assistant Setup..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Load data
echo "📊 Loading data to Neo4j..."
python load_to_neo4j.py

echo "📤 Uploading embeddings to Pinecone..."
python pinecone_upload.py

echo "✅ Setup complete! Starting chat..."
python hybrid_chat.py
```

Run with:
```bash
chmod +x quick_start.sh
./quick_start.sh
```

---

## Production Deployment

### Using Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:latest
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: neo4j/password
    volumes:
      - neo4j_data:/data

  travel-assistant:
    build: .
    depends_on:
      - neo4j
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
    ports:
      - "8000:8000"

volumes:
  neo4j_data:
```

Run with:
```bash
docker-compose up -d
```

---

## Maintenance

### Update Data
```bash
# Add new attractions to vietnam_travel_dataset.json
# Then reload:
python load_to_neo4j.py
python pinecone_upload.py
```

### Clear Cache
```bash
# Restart Python to clear embedding cache
# Or implement cache clear function
```

### Monitor Performance
```bash
# Check Neo4j stats
curl http://localhost:7474/db/data/

# Check Pinecone stats
python -c "from pinecone import Pinecone; pc = Pinecone(api_key='...'); print(pc.Index('vietnam-travel').describe_index_stats())"
```

---

## Support Resources

- **Gemini Docs**: https://ai.google.dev/docs
- **Pinecone Docs**: https://docs.pinecone.io/
- **Neo4j Docs**: https://neo4j.com/docs/
- **Project Documentation**: See improvements.md

---

## Checklist

Before running:
- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Neo4j running
- [ ] API keys configured in config.py
- [ ] Data loaded to Neo4j
- [ ] Embeddings uploaded to Pinecone
- [ ] Test query successful

---

**🎉 You're all set! Happy travels with your AI assistant!**