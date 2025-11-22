# Quick Reference - Communication Scorer

## ⚡ Quick Start Commands

```bash
# Setup (one-time)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run
python app.py

# Test
python test_scorer.py
python api_client.py
```

## 🔧 Common Commands

### Virtual Environment
```bash
# Create
python -m venv venv

# Activate
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows CMD
venv\Scripts\Activate.ps1         # Windows PowerShell

# Deactivate
deactivate

# Delete
rm -rf venv                       # macOS/Linux
rmdir /s venv                     # Windows
```

### Dependencies
```bash
# Install all
pip install -r requirements.txt

# Install specific
pip install flask

# Update all
pip install --upgrade -r requirements.txt

# List installed
pip list

# Freeze to file
pip freeze > requirements.txt
```

### Running the Server
```bash
# Development (auto-reload)
python app.py

# Production
gunicorn app:app

# Production with options
gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 app:app

# Background (Linux/Mac)
nohup gunicorn app:app &

# View background processes
ps aux | grep gunicorn

# Kill process
pkill -f gunicorn
```

## 🧪 Testing Commands

### API Testing (cURL)
```bash
# Health check
curl http://localhost:5000/api/health

# Score transcript
curl -X POST http://localhost:5000/api/score \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Hello everyone..."}'

# Pretty print JSON
curl http://localhost:5000/api/health | python -m json.tool

# Save response to file
curl http://localhost:5000/api/health > response.json
```

### Testing (Python)
```bash
# Run test script
python test_scorer.py

# Run with specific Python
python3 test_scorer.py

# Run API client
python api_client.py

# Run with custom URL
python api_client.py http://your-server:5000
```

## 🐳 Docker Commands

```bash
# Build
docker build -t communication-scorer .

# Run
docker run -p 5000:5000 communication-scorer

# Run detached
docker run -d -p 5000:5000 --name scorer-api communication-scorer

# View logs
docker logs scorer-api
docker logs -f scorer-api  # Follow logs

# Stop
docker stop scorer-api

# Remove
docker rm scorer-api

# Remove image
docker rmi communication-scorer

# Docker Compose
docker-compose up
docker-compose up -d  # Detached
docker-compose down
docker-compose logs -f
```

## 🌐 Deployment Commands

### Git
```bash
# Initialize
git init
git add .
git commit -m "Initial commit"

# Push to GitHub
git remote add origin <repo-url>
git branch -M main
git push -u origin main
```

### Heroku
```bash
# Login
heroku login

# Create app
heroku create app-name

# Deploy
git push heroku main

# View logs
heroku logs --tail

# Scale
heroku ps:scale web=1

# Open
heroku open
```

### AWS EC2
```bash
# Connect
ssh -i "key.pem" ubuntu@ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Start service
sudo systemctl start scorer-api
sudo systemctl stop scorer-api
sudo systemctl restart scorer-api
sudo systemctl status scorer-api

# View logs
sudo journalctl -u scorer-api -f

# Nginx
sudo systemctl restart nginx
sudo nginx -t  # Test config
```

## 🔍 Debugging Commands

### Check Port Usage
```bash
# Linux/Mac
lsof -ti:5000
lsof -ti:5000 | xargs kill -9  # Kill process

# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Check Python/Pip
```bash
# Python version
python --version
python3 --version

# Pip version
pip --version

# Where is Python
which python     # macOS/Linux
where python     # Windows

# Python path
python -c "import sys; print(sys.executable)"
```

### Check Disk Space
```bash
# Linux/Mac
df -h
du -sh *

# Windows
dir
```

### Check Memory
```bash
# Linux/Mac
free -h
top

# Windows
tasklist
wmic memorychip get capacity
```

### View Logs
```bash
# Application logs
tail -f app.log
tail -n 100 app.log  # Last 100 lines

# System logs (Linux)
sudo tail -f /var/log/syslog

# Nginx logs
sudo tail -f /var/log/nginx/error.log
sudo tail -f /var/log/nginx/access.log
```

## 📊 Monitoring Commands

### Check API Status
```bash
# Simple check
curl http://localhost:5000/api/health

# With timing
time curl http://localhost:5000/api/health

# Continuous monitoring
watch -n 5 'curl -s http://localhost:5000/api/health | python -m json.tool'
```

### Performance Testing
```bash
# Apache Bench (install first)
ab -n 100 -c 10 http://localhost:5000/api/health

# Using wrk
wrk -t4 -c100 -d30s http://localhost:5000/api/health
```

## 📝 File Operations

### View Files
```bash
# List files
ls -la              # Linux/Mac
dir                 # Windows

# View file
cat filename
less filename
head -n 10 filename
tail -n 10 filename

# Edit file
nano filename       # Linux/Mac
notepad filename    # Windows
```

### Copy/Move
```bash
# Copy
cp source dest
scp file user@server:path

# Move
mv source dest

# Delete
rm filename
rm -rf directory
```

## 🔐 Permission Commands (Linux/Mac)

```bash
# Make executable
chmod +x script.sh

# Change ownership
sudo chown user:group filename

# Change permissions
chmod 644 filename
chmod 755 directory
```

## 🌍 Environment Variables

```bash
# Set temporarily
export PORT=5000              # Linux/Mac
set PORT=5000                 # Windows CMD
$env:PORT=5000               # Windows PowerShell

# Set permanently (Linux/Mac)
echo 'export PORT=5000' >> ~/.bashrc
source ~/.bashrc

# View
echo $PORT                    # Linux/Mac
echo %PORT%                   # Windows CMD
echo $env:PORT               # Windows PowerShell
```

## 🔄 Process Management

```bash
# Find process
ps aux | grep python
pgrep -f app.py

# Kill by PID
kill PID
kill -9 PID  # Force kill

# Kill by name
pkill -f app.py
killall python
```

## 📦 Package Management

```bash
# Update pip
pip install --upgrade pip

# Install from GitHub
pip install git+https://github.com/user/repo.git

# Uninstall
pip uninstall package-name

# Show package info
pip show package-name

# Check outdated
pip list --outdated
```

## 🛠️ Troubleshooting One-Liners

```bash
# Clear pip cache
pip cache purge

# Reinstall all packages
pip install --force-reinstall -r requirements.txt

# Fix SSL certificate issues
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org package-name

# Fix permissions
sudo chown -R $USER:$USER venv

# Reset git
rm -rf .git && git init
```

## 💡 Useful Aliases (Add to ~/.bashrc or ~/.zshrc)

```bash
alias activate='source venv/bin/activate'
alias runapi='python app.py'
alias testapi='python test_scorer.py'
alias logs='tail -f app.log'
alias status='curl http://localhost:5000/api/health'
```

## 📞 Emergency Commands

```bash
# If server is stuck
killall python
pkill -9 -f app.py

# If port is stuck
lsof -ti:5000 | xargs kill -9

# If out of disk space
docker system prune -a
pip cache purge
rm -rf ~/.cache/*

# If virtual environment is broken
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🎯 Quick Test Snippet

```python
# Quick API test in Python console
import requests
r = requests.get('http://localhost:5000/api/health')
print(r.json())
```

## 📱 Mobile Testing

```bash
# Find your local IP
ifconfig | grep "inet "     # Mac
ip addr show                # Linux
ipconfig                    # Windows

# Test from mobile (replace with your IP)
curl http://192.168.1.X:5000/api/health
```

---

**Pro Tip:** Bookmark this file for quick reference during development! 🚀
