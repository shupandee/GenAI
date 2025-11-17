# 🎬 Movie Script Writer

> A modular AI-powered screenplay writing framework using MCP and OpenAI Function Calling

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io)

## Overview

Movie Script Writer is an intelligent screenplay generation system that combines the power of AI agents with the Model Context Protocol (MCP) and OpenAI's function calling capabilities. Create professional screenplays through modular, domain-specific agents that handle story structure, character development, dialogue, scenes, and formatting.

### Key Features

- **🎭 Specialized Agents** - Five dedicated agents for different aspects of screenplay writing
- **🔌 MCP Protocol** - Standards-compliant Model Context Protocol server
- **🤖 OpenAI Integration** - Seamless function calling with GPT models
- **💻 CLI Support** - Standalone mode requiring no API keys
- **🔧 Extensible** - Easy to add new agents and capabilities
- **🌐 Multi-Client** - Works with Claude Desktop, OpenAI, or any MCP-compatible client

## Architecture

### System Flow

```
User Request
    ↓
OpenAI GPT Model (decides which function to call)
    ↓
ScreenplayAssistant (orchestration)
    ↓
MCP Client (protocol translation)
    ↓
MCP Server (movie_script_writer.py)
    ↓
Specialized Agents (process & generate)
    ↓
Return synthesized screenplay content
```

### Agent Layer

```
┌─────────────────────────────────────────┐
│         Screenplay Agents               │
├─────────────────────────────────────────┤
│ • Story Architect    - Plot & structure │
│ • Character Developer - Characters      │
│ • Dialogue Specialist - Conversations   │
│ • Scene Builder      - Scene details    │
│ • Format Editor      - Industry format  │
└─────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/movie-script-writer.git
   cd movie-script-writer
   ```

2. **Install dependencies**
   ```bash
   pip install openai mcp jsonschema
   ```

3. **Set up OpenAI API key** (optional for OpenAI integration)
   ```bash
   # Unix/Linux/macOS
   export OPENAI_API_KEY="your-api-key-here"
   
   # Windows PowerShell
   $env:OPENAI_API_KEY="your-api-key-here"
   ```

## Project Structure

```
movie-script-writer/
├── movie_script_writer.py        # MCP server + screenplay agents
├── openai_int.py   # OpenAI integration layer
├── README.md                     # Documentation
├── requirements.txt              # Python dependencies
```

## Usage

### 1. Standalone MCP Server

Run the MCP server manually for integration with Claude Desktop or other MCP clients:

```bash
python movie_script_writer.py < /dev/null
```

### 2. CLI Mode (No API Key Required)

Test the system locally without OpenAI:

```bash
python screenplay_openai_client.py
# Select option 1: Test MCP connection
```

### 3. OpenAI Integration

Run with full OpenAI capabilities:

```bash
python screenplay_openai_client.py
# Select option 2: Run demo scenarios
```

### 4. Interactive Chat Mode

Start an interactive session:

```bash
python screenplay_openai_client.py
# Select option 3: Interactive chat mode
```

**Example interaction:**
```
YOU: Create a three-act thriller about a whistleblower inside a biotech firm.

ASSISTANT: [Calls Story Architect Agent]
Creating story structure with thriller elements...

ACT 1 - THE DISCOVERY
INT. BIOTECH LAB - NIGHT
Dr. Sarah Chen discovers unauthorized genetic experiments...
```

## Available Agents

### 📐 Story Architect
Creates plot structures, acts, and narrative arcs.

**Tool:** `create_story_structure`
```python
{
  "genre": "thriller",
  "premise": "A whistleblower discovers corporate secrets",
  "acts": 3
}
```

### 👤 Character Developer
Develops rich, multi-dimensional characters.

**Tool:** `develop_character`
```python
{
  "name": "Sarah Chen",
  "role": "protagonist",
  "traits": ["intelligent", "determined", "conflicted"]
}
```

### 💬 Dialogue Specialist
Crafts authentic, character-driven dialogue.

**Tool:** `craft_dialogue`
```python
{
  "characters": ["Sarah", "Marcus"],
  "scene_context": "Confrontation in the lab",
  "tone": "tense"
}
```

### 🎬 Scene Builder
Constructs detailed scenes with action and description.

**Tool:** `build_scene`
```python
{
  "location": "INT. BIOTECH LAB - NIGHT",
  "characters": ["Sarah"],
  "action": "Sarah discovers the files"
}
```

### 📝 Format Editor
Ensures industry-standard screenplay formatting.

**Tool:** `get_format_guidelines`
```python
{
  "format_type": "feature_film"
}
```

## How It Works

### Agent Processing

Each agent follows a consistent pattern:

```python
class StoryArchitectAgent(BaseAgent):
    async def process(self, input_data):
        # Process input and generate screenplay content
        structure = self._generate_structure(input_data)
        
        return AgentResponse(
            agent_name="Story Architect",
            content=structure,
            metadata={"genre": input_data.get("genre")},
            timestamp=datetime.now().isoformat()
        )
```

### MCP Tool Registration

Tools are automatically registered with the MCP server:

```python
@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="create_story_structure",
            description="Generate screenplay story structure",
            inputSchema={...}
        ),
        # ... more tools
    ]
```

### OpenAI Function Calling

The ScreenplayAssistant converts MCP tools to OpenAI function schemas:

```python
def _convert_mcp_tools_to_openai(self, mcp_tools):
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        }
        for tool in mcp_tools
    ]
```

## Extending the System

### Adding a New Agent

1. Create a new agent class:
```python
class WorldBuilderAgent(BaseAgent):
    async def process(self, input_data):
        world = self._build_world(input_data)
        return AgentResponse(
            agent_name="World Builder",
            content=world,
            metadata={"setting": input_data.get("setting")},
            timestamp=datetime.now().isoformat()
        )
```

2. Register the tool:
```python
@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "build_world":
        agent = WorldBuilderAgent()
        return await agent.process(arguments)
```

3. Add to tool list in `list_tools()`

### Integration Options

- **Claude Desktop**: Configure as an MCP server in `claude_desktop_config.json`
- **Custom Applications**: Use the MCP client to call tools programmatically
- **API Services**: Deploy as a microservice with REST wrapper
- **CLI Tools**: Extend `screenplay_openai_client.py` for custom workflows

## Configuration

### Environment Variables

```bash
# Required for OpenAI integration
OPENAI_API_KEY=your-api-key

# Optional
MCP_SERVER_PORT=3000
LOG_LEVEL=INFO
```

### Claude Desktop Integration

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "screenplay": {
      "command": "python",
      "args": ["/path/to/movie_script_writer.py"],
      "env": {}
    }
  }
}
```

## Examples

### Example 1: Generate a Complete Story Structure

```python
from screenplay_openai_client import ScreenplayAssistant

assistant = ScreenplayAssistant()
response = assistant.chat("Create a sci-fi story about first contact with aliens")
print(response)
```

### Example 2: Develop Multiple Characters

```python
response = assistant.chat(
    "Develop three main characters for a heist movie: "
    "the mastermind, the tech expert, and the driver"
)
```

### Example 3: Write a Complete Scene

```python
response = assistant.chat(
    "Write a tense dialogue scene between a detective "
    "and a suspect in an interrogation room"
)
```

## Troubleshooting

### Common Issues

**MCP Server not responding**
- Ensure Python 3.8+ is installed
- Check that all dependencies are installed
- Verify the server is running: `python movie_script_writer.py`

**OpenAI API errors**
- Confirm your API key is set correctly
- Check your API key has sufficient credits
- Verify internet connectivity

**Import errors**
- Run `pip install -r requirements.txt`
- Ensure you're using the correct Python environment

## Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-agent`)
3. Commit your changes (`git commit -m 'Add new agent'`)
4. Push to the branch (`git push origin feature/amazing-agent`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include unit tests for new agents
- Update documentation for new features

## Roadmap

- [ ] Web interface for screenplay generation
- [ ] LLM-powered sub-agents for advanced creativity
- [ ] Streaming support for real-time generation
- [ ] Export to PDF, Final Draft, and Celtx formats
- [ ] Collaboration features for team writing
- [ ] Integration with more MCP-compatible clients
- [ ] Pre-built screenplay templates
- [ ] Character relationship mapping
- [ ] Theme and motif analysis tools

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Movie Script Writer Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

## Acknowledgments

- Built with [Model Context Protocol (MCP)](https://modelcontextprotocol.io)
- Powered by [OpenAI GPT Models](https://openai.com)
- Inspired by Google ADK agent architecture

## Support

- **Documentation**: [Wiki](https://github.com/yourusername/movie-script-writer/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/movie-script-writer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/movie-script-writer/discussions)
- **Email**: support@moviescriptwriter.dev

## Citation

If you use this project in your research or work, please cite:

```bibtex
@software{movie_script_writer,
  title = {Movie Script Writer: MCP-Based Screenplay Generation},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/movie-script-writer}
}
```

---

<p align="center">
  Made with ❤️ by the Movie Script Writer team
</p>

<p align="center">
  <a href="#overview">Back to Top</a>
</p>
