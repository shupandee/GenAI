#!/usr/bin/env python3
"""
Movie Script Writer Agent - Works standalone, with Claude Desktop, and OpenAI
No API keys required - uses local processing
"""

import asyncio
import sys
import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# AGENT DEFINITIONS (Google ADK Style)
# ============================================================================

@dataclass
class AgentResponse:
    """Response from an agent"""
    agent_name: str
    content: str
    metadata: Dict[str, Any]
    timestamp: str


class BaseAgent:
    """Base agent class"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        raise NotImplementedError


class StoryArchitectAgent(BaseAgent):
    """Designs overall story structure, plot arcs, and narrative flow"""
    
    def __init__(self):
        super().__init__(
            "Story Architect",
            "Creates story structure, plot points, and narrative arcs"
        )
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        genre = input_data.get("genre", "drama")
        theme = input_data.get("theme", "")
        acts = input_data.get("acts", 3)
        
        structure = self._generate_structure(genre, theme, acts)
        
        return AgentResponse(
            agent_name=self.name,
            content=structure,
            metadata={"genre": genre, "acts": acts},
            timestamp=datetime.now().isoformat()
        )
    
    def _generate_structure(self, genre: str, theme: str, acts: int) -> str:
        """Generate a story structure based on classic screenplay format"""
        structures = {
            "action": {
                "act1": "Setup: Introduce hero in ordinary world → Inciting incident threatens status quo → Hero reluctantly accepts challenge",
                "act2": "Confrontation: Rising stakes and obstacles → Midpoint revelation changes everything → All seems lost moment",
                "act3": "Resolution: Final confrontation → Climactic battle → New equilibrium established"
            },
            "drama": {
                "act1": "Establish relationships and normal life → Catalyst disrupts equilibrium → Point of no return",
                "act2": "Complications deepen conflicts → Relationships tested → Crisis point and dark night",
                "act3": "Truth revealed → Emotional climax → Resolution and character growth"
            },
            "comedy": {
                "act1": "Introduction of quirky characters → Misunderstanding or goal established → Commitment to absurd situation",
                "act2": "Escalating complications and humor → Everything goes wrong → Lowest point with comic despair",
                "act3": "Unexpected solution → Comic resolution → Happy ending with twist"
            },
            "thriller": {
                "act1": "Normal world disrupted → Mystery or threat introduced → Investigation begins",
                "act2": "Clues and red herrings → Danger escalates → Major revelation",
                "act3": "Race against time → Confrontation with antagonist → Truth unveiled"
            },
            "horror": {
                "act1": "Peaceful setting → First signs of threat → Denial and disbelief",
                "act2": "Escalating terror → Characters isolated → True nature of threat revealed",
                "act3": "Final confrontation → Survival or sacrifice → Ambiguous resolution"
            },
            "romance": {
                "act1": "Meet cute or reunion → Initial attraction → Obstacle preventing relationship",
                "act2": "Growing closer → Conflict tests relationship → Misunderstanding or betrayal",
                "act3": "Grand gesture → Declaration of love → Happy resolution"
            }
        }
        
        selected = structures.get(genre.lower(), structures["drama"])
        
        output = f"STORY STRUCTURE - {genre.upper()}\n"
        output += f"Theme: {theme}\n\n"
        
        for i in range(1, min(acts + 1, 4)):
            act_key = f"act{i}"
            if act_key in selected:
                output += f"ACT {i}:\n{selected[act_key]}\n\n"
        
        output += "\nBEAT SHEET (Save the Cat Structure):\n"
        output += "• Opening Image (Page 1) - Visual snapshot of before\n"
        output += "• Setup (Pages 1-10) - Introduce world and characters\n"
        output += "• Catalyst (Page 12) - Inciting incident changes everything\n"
        output += "• Debate (Pages 12-25) - Character wrestles with decision\n"
        output += "• Break into Act 2 (Page 25) - Commitment to journey\n"
        output += "• B Story Begins (Page 30) - Secondary plot/relationship\n"
        output += "• Fun and Games (Pages 30-55) - Promise of premise\n"
        output += "• Midpoint (Page 55) - False victory or defeat\n"
        output += "• Bad Guys Close In (Pages 55-75) - Complications mount\n"
        output += "• All Is Lost (Page 75) - Lowest point\n"
        output += "• Dark Night of the Soul (Pages 75-85) - Internal reflection\n"
        output += "• Break into Act 3 (Page 85) - Solution appears\n"
        output += "• Finale (Pages 85-110) - Final confrontation\n"
        output += "• Final Image (Page 110) - Visual snapshot of after\n"
        
        return output


class CharacterDeveloperAgent(BaseAgent):
    """Creates detailed character profiles and arcs"""
    
    def __init__(self):
        super().__init__(
            "Character Developer",
            "Develops character backgrounds, motivations, and arcs"
        )
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        char_type = input_data.get("character_type", "protagonist")
        genre = input_data.get("genre", "drama")
        
        character = self._create_character(char_type, genre)
        
        return AgentResponse(
            agent_name=self.name,
            content=character,
            metadata={"character_type": char_type},
            timestamp=datetime.now().isoformat()
        )
    
    def _create_character(self, char_type: str, genre: str) -> str:
        """Create a detailed character profile"""
        output = f"CHARACTER PROFILE - {char_type.upper()}\n\n"
        
        output += "BASIC INFO:\n"
        output += "• Name: [To be determined based on your story]\n"
        output += "• Age: [Choose appropriate for role and genre]\n"
        output += "• Occupation: [What they do for a living]\n"
        output += "• Physical Description: [Key visual details]\n\n"
        
        output += "PERSONALITY:\n"
        output += "• Core trait: [Dominant characteristic that defines them]\n"
        output += "• Fatal flaw: [Internal weakness causing problems]\n"
        output += "• Greatest strength: [What makes them capable]\n"
        output += "• Deepest fear: [Psychological fear driving behavior]\n"
        output += "• Secret desire: [What they truly want but won't admit]\n\n"
        
        output += "MOTIVATION:\n"
        output += "• External goal: [Concrete objective they pursue]\n"
        output += "• Internal need: [Emotional/psychological growth needed]\n"
        output += "• Stakes: [What happens if they fail]\n"
        output += "• Opposition: [What stands in their way]\n\n"
        
        output += "BACKSTORY:\n"
        output += "• Defining wound: [Formative traumatic experience]\n"
        output += "• Ghost/Past: [History that haunts them]\n"
        output += "• Misbelief: [False worldview they hold]\n"
        output += "• Origin of misbelief: [How they came to believe it]\n\n"
        
        output += "CHARACTER ARC:\n"
        output += "• Starting point: [Who they are at beginning]\n"
        output += "• Catalyst: [What forces change]\n"
        output += "• Transformation: [How they evolve]\n"
        output += "• End point: [Who they become]\n"
        output += "• Arc type: [Positive, negative, or flat]\n\n"
        
        output += "RELATIONSHIPS:\n"
        output += "• Ally/Mentor: [Key supporting character]\n"
        output += "• Antagonist: [Primary opposition]\n"
        output += "• Love interest: [Romantic subplot if applicable]\n"
        output += "• Reflection character: [Shows what they could become]\n\n"
        
        output += "VOICE & MANNERISMS:\n"
        output += "• Speech pattern: [How they talk - formal, casual, etc.]\n"
        output += "• Vocabulary level: [Education reflected in word choice]\n"
        output += "• Physical ticks: [Nervous habits, gestures]\n"
        output += "• Catchphrase/Saying: [Recurring expression if any]\n"
        
        return output


class DialogueSpecialistAgent(BaseAgent):
    """Crafts authentic dialogue and character voice"""
    
    def __init__(self):
        super().__init__(
            "Dialogue Specialist",
            "Creates natural, character-specific dialogue"
        )
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        scene_type = input_data.get("scene_type", "conversation")
        characters = input_data.get("characters", 2)
        tone = input_data.get("tone", "neutral")
        
        dialogue = self._generate_dialogue_guide(scene_type, characters, tone)
        
        return AgentResponse(
            agent_name=self.name,
            content=dialogue,
            metadata={"scene_type": scene_type, "tone": tone},
            timestamp=datetime.now().isoformat()
        )
    
    def _generate_dialogue_guide(self, scene_type: str, characters: int, tone: str) -> str:
        """Generate dialogue writing guidelines and examples"""
        output = f"DIALOGUE GUIDE - {scene_type.upper()}\n"
        output += f"Tone: {tone} | Characters: {characters}\n\n"
        
        output += "DIALOGUE PRINCIPLES:\n"
        output += "• Subtext: Characters rarely say exactly what they mean\n"
        output += "• Conflict: Every exchange should have tension or stakes\n"
        output += "• Voice: Each character needs distinct speech patterns\n"
        output += "• Economy: Cut unnecessary words; dialogue should be tight\n"
        output += "• Rhythm: Vary sentence length for natural flow\n\n"
        
        output += "EXAMPLE FORMAT:\n\n"
        output += "INT. COFFEE SHOP - DAY\n\n"
        output += "                    SARAH\n"
        output += "          You came.\n\n"
        output += "                    JAMES\n"
        output += "          I almost didn't.\n\n"
        
        return output


class SceneBuilderAgent(BaseAgent):
    """Constructs detailed scene breakdowns"""
    
    def __init__(self):
        super().__init__(
            "Scene Builder",
            "Creates scene-by-scene breakdowns with descriptions"
        )
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        act = input_data.get("act", 1)
        location = input_data.get("location", "interior")
        time = input_data.get("time", "day")
        
        scene = self._create_scene_template(act, location, time)
        
        return AgentResponse(
            agent_name=self.name,
            content=scene,
            metadata={"act": act, "location": location},
            timestamp=datetime.now().isoformat()
        )
    
    def _create_scene_template(self, act: int, location: str, time: str) -> str:
        """Create a scene template in proper screenplay format"""
        output = f"SCENE TEMPLATE - ACT {act}\n\n"
        output += f"INT./EXT. LOCATION - {time.upper()}\n\n"
        output += "Scene description in present tense.\n\n"
        output += "                    CHARACTER NAME\n"
        output += "          Dialogue here.\n\n"
        return output


class FormatEditorAgent(BaseAgent):
    """Ensures proper screenplay formatting"""
    
    def __init__(self):
        super().__init__(
            "Format Editor",
            "Validates and formats screenplay according to industry standards"
        )
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        guidelines = "SCREENPLAY FORMATTING STANDARDS\n\n"
        guidelines += "• Page size: 8.5\" x 11\"\n"
        guidelines += "• Font: 12pt Courier\n"
        guidelines += "• One page ≈ one minute of screen time\n"
        
        return AgentResponse(
            agent_name=self.name,
            content=guidelines,
            metadata={"format": "standard"},
            timestamp=datetime.now().isoformat()
        )


# ============================================================================
# ORCHESTRATOR
# ============================================================================

class MovieScriptOrchestrator:
    """Main orchestrator that coordinates all sub-agents"""
    
    def __init__(self):
        self.agents = {
            "story_architect": StoryArchitectAgent(),
            "character_developer": CharacterDeveloperAgent(),
            "dialogue_specialist": DialogueSpecialistAgent(),
            "scene_builder": SceneBuilderAgent(),
            "format_editor": FormatEditorAgent()
        }
    
    async def get_agent_response(self, agent_name: str, params: Dict[str, Any]) -> AgentResponse:
        """Get response from a specific agent"""
        if agent_name in self.agents:
            return await self.agents[agent_name].process(params)
        raise ValueError(f"Unknown agent: {agent_name}")


# ============================================================================
# MCP SERVER MODE
# ============================================================================

async def run_mcp_server():
    """Run as MCP server for Claude Desktop"""
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent
    except ImportError:
        logger.error("MCP not installed. Run: pip install mcp")
        return

    orchestrator = MovieScriptOrchestrator()
    server = Server("movie-script-writer")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="create_story_structure",
                description="Generate story structure with acts and beat sheet",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "genre": {"type": "string", "description": "Genre (action, drama, comedy, thriller, horror, romance)"},
                        "theme": {"type": "string", "description": "Central theme"},
                        "acts": {"type": "number", "default": 3}
                    }
                }
            ),
            Tool(
                name="develop_character",
                description="Create detailed character profile",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "character_type": {"type": "string", "description": "Type (protagonist, antagonist, etc.)"},
                        "genre": {"type": "string"}
                    }
                }
            ),
            Tool(
                name="craft_dialogue",
                description="Generate dialogue guidelines",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "scene_type": {"type": "string"},
                        "tone": {"type": "string"},
                        "characters": {"type": "number", "default": 2}
                    }
                }
            ),
            Tool(
                name="build_scene",
                description="Create scene template",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "act": {"type": "number", "default": 1},
                        "location": {"type": "string"},
                        "time": {"type": "string"}
                    }
                }
            ),
            Tool(
                name="get_format_guidelines",
                description="Get screenplay formatting standards",
                inputSchema={"type": "object", "properties": {}}
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Any) -> list[TextContent]:
        try:
            agent_map = {
                "create_story_structure": "story_architect",
                "develop_character": "character_developer",
                "craft_dialogue": "dialogue_specialist",
                "build_scene": "scene_builder",
                "get_format_guidelines": "format_editor"
            }
            
            agent_name = agent_map.get(name)
            if not agent_name:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
            
            response = await orchestrator.get_agent_response(agent_name, arguments or {})
            return [TextContent(type="text", text=f"{response.agent_name}:\n\n{response.content}")]
        
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


# ============================================================================
# STANDALONE/CLI MODE
# ============================================================================

async def run_standalone():
    """Run in interactive CLI mode"""
    orchestrator = MovieScriptOrchestrator()
    
    print("\n" + "="*70)
    print("🎬 MOVIE SCRIPT WRITER - Standalone Mode")
    print("="*70)
    print("\nAvailable Commands:")
    print("  1. story    - Create story structure")
    print("  2. character - Develop character")
    print("  3. dialogue  - Craft dialogue")
    print("  4. scene     - Build scene")
    print("  5. format    - Get formatting guidelines")
    print("  6. quit      - Exit")
    print("\n" + "="*70 + "\n")
    
    while True:
        try:
            command = input("Enter command (1-6 or name): ").strip().lower()
            
            if command in ['6', 'quit', 'exit', 'q']:
                print("\n👋 Happy screenwriting!\n")
                break
            
            if command in ['1', 'story']:
                genre = input("Genre (action/drama/comedy/thriller/horror/romance): ").strip() or "drama"
                theme = input("Theme: ").strip() or "redemption"
                acts = input("Number of acts (default 3): ").strip() or "3"
                
                response = await orchestrator.get_agent_response(
                    "story_architect",
                    {"genre": genre, "theme": theme, "acts": int(acts)}
                )
                print(f"\n{response.content}\n")
            
            elif command in ['2', 'character']:
                char_type = input("Character type (protagonist/antagonist/mentor): ").strip() or "protagonist"
                genre = input("Genre: ").strip() or "drama"
                
                response = await orchestrator.get_agent_response(
                    "character_developer",
                    {"character_type": char_type, "genre": genre}
                )
                print(f"\n{response.content}\n")
            
            elif command in ['3', 'dialogue']:
                scene_type = input("Scene type (conversation/confrontation/revelation): ").strip() or "conversation"
                tone = input("Tone (neutral/tense/dramatic): ").strip() or "neutral"
                
                response = await orchestrator.get_agent_response(
                    "dialogue_specialist",
                    {"scene_type": scene_type, "tone": tone, "characters": 2}
                )
                print(f"\n{response.content}\n")
            
            elif command in ['4', 'scene']:
                act = input("Act number (1/2/3): ").strip() or "1"
                location = input("Location (interior/exterior): ").strip() or "interior"
                time = input("Time (day/night): ").strip() or "day"
                
                response = await orchestrator.get_agent_response(
                    "scene_builder",
                    {"act": int(act), "location": location, "time": time}
                )
                print(f"\n{response.content}\n")
            
            elif command in ['5', 'format']:
                response = await orchestrator.get_agent_response("format_editor", {})
                print(f"\n{response.content}\n")
            
            else:
                print("❌ Invalid command. Try again.\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point - auto-detect mode"""
    import sys
    
    # Check if running in MCP mode (stdin is not a terminal)
    if not sys.stdin.isatty():
        logger.info("Starting in MCP server mode...")
        asyncio.run(run_mcp_server())
    else:
        logger.info("Starting in standalone CLI mode...")
        asyncio.run(run_standalone())


if __name__ == "__main__":
    main()