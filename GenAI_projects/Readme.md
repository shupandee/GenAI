# 🤖 AI-Powered Weather Agent

![Project Icon](https://img.shields.io/badge/Project-WeatherAgent-blue)
![Language](https://img.shields.io/badge/Python-3.x-blue)
![Framework](https://img.shields.io/badge/LangChain-v0.1-green)
![LLM](https://img.shields.io/badge/LLM-ChatOpenAI-red)
---

## 📄 Project Overview

This project implements a conversational AI agent that can intelligently answer questions by dynamically using a set of specialized tools. The agent is built using the **LangChain** framework and is designed to handle queries that require a combination of general knowledge and real-time data, specifically focusing on geography and weather.

The core functionality of the agent is to:
- Understand natural language queries.
- Decide which tool to use (web search or a weather API) to find the necessary information.
- Execute the tools to gather data.
- Synthesize the gathered information to provide a comprehensive and accurate answer to the user's question.

The project demonstrates a key application of **agentic AI**, where an LLM is empowered to "think" and "act" by utilizing external resources.

---

## 🛠️ Technical Stack

- **Framework:** `LangChain` (Agent Executor, ReAct)
- **AI Models:** `ChatOpenAI` (as the core LLM), `OpenAIEmbeddings`
- **Tools:**
    - `DuckDuckGoSearchRun`: Used for performing web searches to acquire general knowledge (e.g., finding the capital of a state).
    - `get_weather_data`: A custom Python tool integrated via `@tool` decorator to fetch real-time weather information from `weatherstack.com`.
- **Database:** SQLite3 (for session/user management)
- **Dependencies:** `openai`, `duckduckgo-search`, `requests`, `python-dotenv`, `fastapi` (for API layer)

---

## 🧠 Workflow

The agent's workflow follows a "**Reasoning and Acting**" (ReAct) paradigm. When a user asks a question, the agent performs the following steps:

1.  **Thought:** The LLM analyzes the user's query and determines the next logical step.
2.  **Action:** The LLM selects the most appropriate tool from its available set (e.g., `DuckDuckGoSearch` for a web query or `get_weather_data` for a weather query).
3.  **Observation:** The agent executes the tool and receives a result (e.g., the capital of a state or the current temperature).
4.  **Synthesis:** The LLM combines the observation with its internal knowledge to formulate a final, coherent response.

This process is repeated until the query is fully answered. A visual representation of this flow is shown below:

```plaintext
          ┌───────────────────────┐
          │     User Query        │
          │ (e.g., "Weather in...?")│
          └──────────┬────────────┘
                     │
          ┌──────────▼────────────┐
          │  LangChain Agent      │
          │ (ReAct Agent Logic)   │
          └──────────┬────────────┘
                     │
         (Decision: Which tool to use?)
         ┌──────────┴──────────┐
         │                     │
  ┌──────▼──────┐      ┌──────▼──────┐
  │   Tool:     │      │   Tool:     │
  │ DuckDuckGo  │      │ get_weather │
  │ (Search)    │      │ (Weather API)│
  └──────┬──────┘      └──────┬──────┘
         │                     │
  ┌──────▼──────┐      ┌──────▼──────┐
  │   Search    │      │  Weather    │
  │  for city   │      │   data      │
  │  capital    │      │ (Temperature)│
  └──────┬──────┘      └──────┬──────┘
         └──────────┬────────────┘
                    │
           ┌────────▼─────────┐
           │LLM Synthesizes   │
           │(Combines tool    │
           │outputs & answers)│
           └────────┬─────────┘
                    │
           ┌────────▼─────────┐
           │ Final Answer     │
           │(e.g., "The capital is...│
           │ and the weather is...") │
           └──────────────────┘
