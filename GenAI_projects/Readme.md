# 🌦️ 🤖 AI-Powered Weather Agent  

![Project](https://img.shields.io/badge/Project-WeatherAgent-blue?style=for-the-badge&logo=github)  
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)  
![LangChain](https://img.shields.io/badge/LangChain-v0.1-0E83CD?style=for-the-badge&logo=chainlink&logoColor=white)  
![OpenAI](https://img.shields.io/badge/LLM-ChatOpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)  
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)  
![SQLite](https://img.shields.io/badge/Database-SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)  

---

## 📌 Overview  

The **AI-Powered Weather Agent** is a conversational assistant built with **LangChain** ⚡ and powered by **LLMs** 🧠.  
It can **reason, decide, and act** — intelligently combining general knowledge with **real-time weather data**.  

✨ **Key Capabilities:**  
- 🗣️ Understands natural language queries.  
- 🔍 Chooses the right tool (Search or Weather API).  
- 🌍 Fetches real-time weather from `weatherstack.com`.  
- 🧩 Synthesizes results into a clear, human-like response.  

This project demonstrates **agentic AI**, where an LLM is empowered to take actions using external tools.  

---

## 🛠️ Tech Stack  

| Component       | Technology Used |
|-----------------|-----------------|
| **Framework**   | ⚡ [LangChain](https://www.langchain.com/) (ReAct paradigm) |
| **LLM**         | 🧠 [OpenAI Chat Models](https://platform.openai.com/) (`ChatOpenAI`, `OpenAIEmbeddings`) |
| **Tools**       | 🔍 `DuckDuckGoSearchRun` (web search)<br>☁️ `get_weather_data` (custom Weather API tool) |
| **Database**    | 🗄️ [SQLite3](https://www.sqlite.org/) |
| **API Layer**   | 🚀 [FastAPI](https://fastapi.tiangolo.com/) |
| **Dependencies**| 🐍 `openai`, `duckduckgo-search`, `requests`, `python-dotenv`, `fastapi` |

---

## 🔄 Workflow  

The agent uses the **ReAct pattern** (Reasoning + Acting).  

### Steps:
1. 💭 **Thought** → Analyze user query.  
2. ⚡ **Action** → Select tool (`Search` or `Weather API`).  
3. 👀 **Observation** → Execute and fetch results.  
4. 🧠 **Synthesis** → Combine results + LLM knowledge.  
5. ✅ **Final Answer** → Deliver to user.  

---

## 📊 Visual Flow  

```mermaid
flowchart TD
    A[💬 User Query] --> B[🧠 LangChain Agent (ReAct)]
    B --> C{Which tool to use?}
    C -->|Search Info| D[🔍 DuckDuckGoSearch]
    C -->|Weather Data| E[🌦️ get_weather_data API]

    D --> F[📍 City / Location Info]
    E --> G[🌡️ Temperature / Weather Data]

    F --> H[🧩 LLM Synthesizes Answer]
    G --> H[🧩 LLM Synthesizes Answer]

    H --> I[✅ Final Answer Delivered]
