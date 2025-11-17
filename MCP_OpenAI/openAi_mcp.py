#!/usr/bin/env python3
"""
OpenAI Integration with Movie Script Writer MCP Server
Uses function calling to integrate screenplay agents with OpenAI
"""

import asyncio
import json
import os
from typing import Optional

# Check for OpenAI
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI not installed. Run: pip install openai")

# Check for MCP
try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️  MCP not installed. Run: pip install mcp")


class ScreenplayAssistant:
    """Screenplay assistant that integrates MCP agents with OpenAI"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key and OPENAI_AVAILABLE:
            print("⚠️  Warning: No OpenAI API key found!")
            print("   Set with: export OPENAI_API_KEY='your-key'")
        
        if OPENAI_AVAILABLE:
            self.client = AsyncOpenAI(api_key=self.api_key)
        else:
            self.client = None
    
    async def call_screenplay_agent(self, tool_name: str, arguments: dict, session) -> str:
        """Call an MCP tool and return the response"""
        try:
            result = await session.call_tool(tool_name, arguments)
            # Handle different response formats
            if hasattr(result, 'content'):
                if isinstance(result.content, list) and len(result.content) > 0:
                    if hasattr(result.content[0], 'text'):
                        return result.content[0].text
                    return str(result.content[0])
                return str(result.content)
            return str(result)
        except Exception as e:
            return f"Error calling agent: {str(e)}"
    
    async def chat_with_agents(self, user_message: str, verbose: bool = True):
        """
        Chat with OpenAI using screenplay agents via MCP
        """
        if not OPENAI_AVAILABLE:
            print("❌ OpenAI package not installed. Install with: pip install openai")
            return None
        
        if not MCP_AVAILABLE:
            print("❌ MCP package not installed. Install with: pip install mcp")
            return None
        
        if not self.api_key:
            print("❌ OpenAI API key not set. Set with: export OPENAI_API_KEY='your-key'")
            return None
        
        # Connect to MCP server
        server_params = StdioServerParameters(
            command="python",
            args=["movie_script_writer.py"],
            env=None
        )
        
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize MCP session
                    await session.initialize()
                    
                    # Get available tools from MCP
                    tools_result = await session.list_tools()
                    
                    # Handle different response formats
                    if hasattr(tools_result, 'tools'):
                        mcp_tools = tools_result.tools
                    else:
                        mcp_tools = tools_result
                    
                    # Convert MCP tools to OpenAI function format
                    openai_functions = []
                    for tool in mcp_tools:
                        func_def = {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                            }
                        }
                        
                        # Add input schema if available
                        if hasattr(tool, 'inputSchema'):
                            func_def["function"]["parameters"] = tool.inputSchema
                        
                        openai_functions.append(func_def)
                    
                    if verbose:
                        print(f"\n{'='*70}")
                        print(f"USER: {user_message}")
                        print(f"{'='*70}\n")
                    
                    # Create chat completion with function calling
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a professional screenplay writing assistant. "
                                       "You help writers create compelling scripts by using specialized "
                                       "agents for story structure, character development, dialogue, "
                                       "scene building, and formatting guidance. Always use the available "
                                       "tools when they can help answer the user's question."
                        },
                        {
                            "role": "user",
                            "content": user_message
                        }
                    ]
                    
                    # First API call - OpenAI decides which tools to use
                    response = await self.client.chat.completions.create(
                        model="gpt-4o-mini",  # Using mini for cost efficiency
                        messages=messages,
                        tools=openai_functions,
                        tool_choice="auto"
                    )
                    
                    response_message = response.choices[0].message
                    
                    # Check if OpenAI wants to call any tools
                    if response_message.tool_calls:
                        if verbose:
                            print("🔧 OpenAI is calling screenplay agents...\n")
                        
                        # Add assistant's response to messages
                        messages.append({
                            "role": "assistant",
                            "content": response_message.content,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments
                                    }
                                }
                                for tc in response_message.tool_calls
                            ]
                        })
                        
                        # Execute each tool call via MCP
                        for tool_call in response_message.tool_calls:
                            function_name = tool_call.function.name
                            function_args = json.loads(tool_call.function.arguments)
                            
                            if verbose:
                                print(f"   📝 Calling: {function_name}")
                                print(f"   📋 Arguments: {json.dumps(function_args, indent=2)}\n")
                            
                            # Call the MCP tool
                            tool_response = await self.call_screenplay_agent(
                                function_name,
                                function_args,
                                session
                            )
                            
                            # Add tool response to messages
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": function_name,
                                "content": tool_response
                            })
                        
                        # Second API call - get final response with tool results
                        if verbose:
                            print("🤖 Generating final response...\n")
                        
                        final_response = await self.client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=messages
                        )
                        
                        final_message = final_response.choices[0].message.content
                    else:
                        # No tool calls needed
                        final_message = response_message.content
                    
                    if verbose:
                        print(f"{'='*70}")
                        print(f"ASSISTANT:\n\n{final_message}")
                        print(f"{'='*70}\n")
                    
                    return final_message
        
        except FileNotFoundError:
            print("❌ Error: Could not find movie_script_writer.py")
            print("   Make sure the script is in the same directory or update the path.")
            return None
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


async def demo_scenarios():
    """Run demo scenarios showing OpenAI + MCP integration"""
    
    print("\n" + "🎬"*35)
    print("MOVIE SCRIPT WRITER - OpenAI Integration Demo")
    print("🎬"*35 + "\n")
    
    assistant = ScreenplayAssistant()
    
    scenarios = [
        "I want to write a thriller about corporate espionage. Help me create a story structure.",
        "Develop a complex antagonist character for a psychological thriller.",
        "I need a tense confrontation scene between a detective and a suspect. Give me dialogue guidelines.",
        "What are the proper screenplay formatting guidelines?"
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'#'*70}")
        print(f"SCENARIO {i}/{len(scenarios)}")
        print(f"{'#'*70}\n")
        
        await assistant.chat_with_agents(scenario)
        
        if i < len(scenarios):
            print("\nPress Enter for next scenario...")
            input()


async def interactive_mode():
    """Interactive chat mode with OpenAI"""
    
    print("\n" + "🎬"*35)
    print("MOVIE SCRIPT WRITER - Interactive Mode")
    print("🎬"*35)
    print("\nChat with the AI screenplay assistant!")
    print("Type 'quit' to exit.\n")
    print("="*70 + "\n")
    
    assistant = ScreenplayAssistant()
    
    while True:
        try:
            user_input = input("YOU: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Happy screenwriting!\n")
                break
            
            if not user_input:
                continue
            
            await assistant.chat_with_agents(user_input)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


async def simple_test():
    """Simple test without OpenAI - just tests MCP connection"""
    print("\n" + "🎬"*35)
    print("TESTING MCP CONNECTION (No OpenAI needed)")
    print("🎬"*35 + "\n")
    
    if not MCP_AVAILABLE:
        print("❌ MCP package not installed. Install with: pip install mcp")
        return
    
    server_params = StdioServerParameters(
        command="python",
        args=["movie_script_writer.py"],
        env=None
    )
    
    try:
        print("Connecting to MCP server...")
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("✅ Connected!\n")
                
                # List tools
                tools_result = await session.list_tools()
                if hasattr(tools_result, 'tools'):
                    tools = tools_result.tools
                else:
                    tools = tools_result
                
                print(f"Found {len(tools)} tools:")
                for tool in tools:
                    print(f"  • {tool.name}: {tool.description}")
                
                # Test a tool call
                print("\n" + "="*70)
                print("Testing: create_story_structure")
                print("="*70 + "\n")
                
                result = await session.call_tool(
                    "create_story_structure",
                    {"genre": "thriller", "theme": "redemption", "acts": 3}
                )
                
                # Handle different response formats
                if hasattr(result, 'content'):
                    if isinstance(result.content, list) and len(result.content) > 0:
                        if hasattr(result.content[0], 'text'):
                            content = result.content[0].text
                        else:
                            content = str(result.content[0])
                    else:
                        content = str(result.content)
                else:
                    content = str(result)
                
                print(content[:500] + "..." if len(content) > 500 else content)
                print("\n✅ MCP connection working perfectly!\n")
    
    except FileNotFoundError:
        print("❌ Error: Could not find movie_script_writer.py")
        print("   Make sure the script is in the same directory.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point"""
    
    if not MCP_AVAILABLE:
        print("\n❌ MCP package not installed!")
        print("Install with: pip install mcp\n")
        return
    
    print("\n🎬 Movie Script Writer - OpenAI Integration")
    print("="*70)
    print("\nSelect mode:")
    print("1. Test MCP connection (no OpenAI needed)")
    print("2. Demo scenarios (requires OpenAI API key)")
    print("3. Interactive chat mode (requires OpenAI API key)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        asyncio.run(simple_test())
    elif choice == '2':
        if not OPENAI_AVAILABLE:
            print("\n❌ OpenAI package not installed!")
            print("Install with: pip install openai\n")
        else:
            asyncio.run(demo_scenarios())
    elif choice == '3':
        if not OPENAI_AVAILABLE:
            print("\n❌ OpenAI package not installed!")
            print("Install with: pip install openai\n")
        else:
            asyncio.run(interactive_mode())
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()