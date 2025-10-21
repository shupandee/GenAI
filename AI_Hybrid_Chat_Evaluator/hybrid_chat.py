# hybrid_chat.py
"""
Hybrid AI Travel Assistant - Complete Version
Pinecone + Neo4j + Gemini AI (with intelligent fallback system)
"""
import os
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

from typing import List
import re
from collections import defaultdict
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import config

# -----------------------------
# Configuration
# -----------------------------
EMBED_MODEL = "models/text-embedding-004"
CHAT_MODEL = config.GEMINI_CHAT_MODEL
TOP_K = 5
INDEX_NAME = config.PINECONE_INDEX_NAME

# -----------------------------
# Initialize Clients
# -----------------------------
genai.configure(api_key=config.GEMINI_API_KEY)
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# Connect to Pinecone index
if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    print(f"Creating serverless index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=config.EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# Connect to Neo4j
driver = GraphDatabase.driver(
    config.NEO4J_URI, 
    auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)

# -----------------------------
# Helper Functions
# -----------------------------
def embed_text(text: str) -> List[float]:
    """Generate embedding using Gemini"""
    try:
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return []

def pinecone_query(query_text: str, top_k=TOP_K):
    """Query Pinecone index using embedding"""
    vec = embed_text(query_text)
    if not vec:
        print("⚠️  Failed to generate embedding")
        return []
    
    res = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )
    
    print(f"🔍 Pinecone Results: {len(res.matches)} matches found")
    return res.matches

def fetch_graph_context(node_ids: List[str]):
    """
    Fetch neighboring nodes from Neo4j
    Updated to use Entity label
    """
    if not node_ids:
        return []
    
    facts = []
    with driver.session() as session:
        query = """
        MATCH (n:Entity) WHERE n.id IN $node_ids
        OPTIONAL MATCH (n)-[r]-(m:Entity)
        RETURN n.id AS source_id,
               n.name AS source_name,
               type(r) AS rel, 
               m.id AS id, 
               m.name AS name, 
               m.type AS type, 
               m.description AS description,
               m.region AS region,
               m.tags AS tags
        LIMIT 30
        """
        try:
            results = session.run(query, node_ids=node_ids[:5])
            for record in results:
                if record["rel"]:  # Only add if relationship exists
                    facts.append({
                        "source": record["source_id"],
                        "source_name": record["source_name"],
                        "rel": record["rel"],
                        "target_id": record["id"],
                        "target_name": record["name"],
                        "target_type": record["type"],
                        "target_desc": (record["description"] or "")[:300],
                        "target_region": record.get("region", ""),
                        "target_tags": record.get("tags", [])
                    })
        except Exception as e:
            print(f"⚠️  Neo4j query error: {e}")
    
    print(f"🕸️  Graph Context: {len(facts)} relationships found")
    return facts

def build_prompt(user_query, pinecone_matches, graph_facts):
    """Build a chat prompt combining vector DB matches and graph facts"""
    
    # Build vector search context
    vec_context = []
    for i, m in enumerate(pinecone_matches[:5], 1):
        meta = m.metadata
        snippet = f"{i}. **{meta.get('name', 'Unknown')}** ({meta.get('type', 'N/A')})\n"
        
        if meta.get('description'):
            snippet += f"   Description: {meta['description'][:150]}\n"
        
        details = []
        if meta.get('region'):
            details.append(f"Region: {meta['region']}")
        if meta.get('duration'):
            details.append(f"Duration: {meta['duration']}")
        if meta.get('cost'):
            details.append(f"Cost: {meta['cost']}")
        
        if details:
            snippet += f"   {' | '.join(details)}\n"
        
        vec_context.append(snippet)
    
    # Build graph context
    graph_context = []
    if graph_facts:
        seen = set()
        for fact in graph_facts[:15]:
            target = fact.get('target_name')
            if target and target not in seen:
                seen.add(target)
                graph_context.append(f"- {target} ({fact.get('target_type', 'N/A')})")
    
    # Construct simple, safe prompt
    prompt = f"""You are a Vietnam travel expert. Answer this query: {user_query}

Available options:

{chr(10).join(vec_context)}

Related attractions:
{chr(10).join(graph_context[:10]) if graph_context else "No additional connections."}

Provide helpful recommendations with:
1. Specific place names from the list
2. Brief descriptions
3. Practical travel tips
4. Clear, organized format

Keep response positive and concise (max 300 words)."""
    
    return prompt

def format_itinerary(query: str, matches: list, graph_facts: list, num_days: int, is_romantic: bool) -> str:
    """Format as a day-by-day itinerary"""
    
    response = f"# 🎒 Your {num_days}-Day Vietnam Itinerary\n\n"
    
    if is_romantic:
        response += "💕 *Specially curated for couples*\n\n"
    
    response += f"**Based on your query:** *{query}*\n\n"
    response += "---\n\n"
    
    # Group matches by city/region
    locations_by_region = defaultdict(list)
    
    for match in matches:
        meta = match.metadata
        region = meta.get('region', 'Vietnam')
        location_type = meta.get('type', 'Unknown')
        
        locations_by_region[region].append({
            'name': meta.get('name', 'Unknown'),
            'type': location_type,
            'description': meta.get('description', ''),
            'duration': meta.get('duration', ''),
            'cost': meta.get('cost', ''),
            'tags': meta.get('tags', ''),
            'score': match.score
        })
    
    # Sort locations by score
    for region in locations_by_region:
        locations_by_region[region].sort(key=lambda x: x['score'], reverse=True)
    
    # Create day-by-day itinerary
    day_counter = 1
    activities_assigned = 0
    
    for region, locations in locations_by_region.items():
        if day_counter > num_days:
            break
            
        # Find cities in this region
        cities = [loc for loc in locations if loc['type'] == 'City']
        activities = [loc for loc in locations if loc['type'] in ['Activity', 'Attraction']]
        hotels = [loc for loc in locations if loc['type'] == 'Hotel']
        
        if cities and day_counter <= num_days:
            city = cities[0]
            response += f"## 📍 Day {day_counter}"
            
            if num_days > 1 and day_counter < num_days:
                response += f"-{min(day_counter + 1, num_days)}"
            
            response += f": {city['name']}\n\n"
            
            # City overview
            if city['description']:
                desc = city['description'][:200]
                response += f"*{desc}{'...' if len(city['description']) > 200 else ''}*\n\n"
            
            response += "### 🌅 Morning\n"
            if activities:
                morning_activity = activities[0] if len(activities) > 0 else None
                if morning_activity:
                    response += f"**{morning_activity['name']}**\n"
                    if morning_activity['description']:
                        response += f"{morning_activity['description'][:150]}...\n"
                    if morning_activity['duration']:
                        response += f"⏱️ Duration: {morning_activity['duration']}\n"
                    if morning_activity['cost']:
                        response += f"💰 Cost: {morning_activity['cost']}\n"
                    response += "\n"
            
            response += "### 🌞 Afternoon\n"
            if len(activities) > 1:
                afternoon_activity = activities[1]
                response += f"**{afternoon_activity['name']}**\n"
                if afternoon_activity['description']:
                    response += f"{afternoon_activity['description'][:150]}...\n"
                if afternoon_activity['duration']:
                    response += f"⏱️ Duration: {afternoon_activity['duration']}\n"
                if afternoon_activity['cost']:
                    response += f"💰 Cost: {afternoon_activity['cost']}\n"
                response += "\n"
            
            response += "### 🌙 Evening\n"
            if len(activities) > 2:
                evening_activity = activities[2]
                response += f"**{evening_activity['name']}**\n"
                if evening_activity['description']:
                    response += f"{evening_activity['description'][:150]}...\n"
            else:
                response += "Enjoy a leisurely dinner at a local restaurant\n"
            response += "\n"
            
            # Add hotel recommendation
            if hotels:
                response += "### 🏨 Where to Stay\n"
                hotel = hotels[0]
                response += f"**{hotel['name']}**\n"
                if hotel['description']:
                    response += f"{hotel['description'][:100]}...\n"
                if hotel['cost']:
                    response += f"💰 {hotel['cost']}\n"
                response += "\n"
            
            response += "---\n\n"
            day_counter += 2 if num_days > 3 else 1
            activities_assigned += len(activities[:3])
    
    # Add practical information
    response += "## 💡 Travel Tips\n\n"
    
    # Extract unique regions
    regions = list(locations_by_region.keys())
    if regions:
        response += f"- **Regions covered:** {', '.join(regions[:3])}\n"
    
    # Add best time info from matches
    best_times = set()
    for match in matches[:5]:
        bt = match.metadata.get('best_time_to_visit')
        if bt:
            best_times.add(bt)
    
    if best_times:
        response += f"- **Best time to visit:** {', '.join(list(best_times)[:2])}\n"
    
    response += f"- **Estimated activities:** {activities_assigned} unique experiences\n"
    
    # Add romantic tips if applicable
    if is_romantic:
        response += "\n### 💕 Romantic Touches\n"
        response += "- Book sunset activities in advance\n"
        response += "- Consider private tours for intimate experiences\n"
        response += "- Try couples' spa treatments\n"
        response += "- Reserve window seats at restaurants\n"
    
    # Add nearby attractions from graph
    if graph_facts:
        response += "\n## 🔗 Also Consider Nearby\n\n"
        seen = set()
        count = 0
        for fact in graph_facts[:10]:
            target = fact.get('target_name')
            if target and target not in seen and count < 5:
                seen.add(target)
                response += f"- {target} ({fact.get('target_type', 'N/A')})\n"
                count += 1
    
    response += "\n---\n\n"
    response += "✨ *This itinerary is customized based on your preferences. Adjust timing based on your pace!*\n"
    
    return response

def format_recommendations(query: str, matches: list, graph_facts: list) -> str:
    """Format as a list of recommendations (non-itinerary queries)"""
    
    response = f"# 🎯 Recommendations for: {query}\n\n"
    
    # Group by type
    by_type = defaultdict(list)
    
    for match in matches:
        meta = match.metadata
        location_type = meta.get('type', 'Other')
        by_type[location_type].append({
            'name': meta.get('name', 'Unknown'),
            'description': meta.get('description', ''),
            'region': meta.get('region', ''),
            'duration': meta.get('duration', ''),
            'cost': meta.get('cost', ''),
            'best_time': meta.get('best_time_to_visit', ''),
            'tags': meta.get('tags', ''),
            'score': match.score
        })
    
    # Display by category
    type_icons = {
        'City': '🏙️',
        'Activity': '🎯',
        'Attraction': '🎨',
        'Hotel': '🏨',
        'Restaurant': '🍽️',
        'Beach': '🏖️'
    }
    
    for location_type, items in sorted(by_type.items(), key=lambda x: len(x[1]), reverse=True):
        icon = type_icons.get(location_type, '📍')
        response += f"## {icon} {location_type}s\n\n"
        
        for i, item in enumerate(items[:3], 1):
            response += f"### {i}. {item['name']}\n\n"
            
            if item['description']:
                desc = item['description']
                if len(desc) > 200:
                    desc = desc[:200] + "..."
                response += f"{desc}\n\n"
            
            # Add details in a clean format
            details = []
            if item['region']:
                details.append(f"📍 {item['region']}")
            if item['duration']:
                details.append(f"⏱️ {item['duration']}")
            if item['cost']:
                details.append(f"💰 {item['cost']}")
            if item['best_time']:
                details.append(f"📅 Best: {item['best_time']}")
            
            if details:
                response += " • ".join(details) + "\n\n"
            
            # Add tags if available
            if item['tags']:
                tags = item['tags'].split(',') if isinstance(item['tags'], str) else item['tags']
                if tags:
                    response += f"🏷️ {', '.join(tags[:5])}\n\n"
            
            response += f"*Match score: {item['score']:.1%}*\n\n"
            response += "---\n\n"
    
    # Add nearby attractions
    if graph_facts:
        response += "## 🔗 Nearby & Related\n\n"
        seen = set()
        for fact in graph_facts[:12]:
            target = fact.get('target_name')
            if target and target not in seen:
                seen.add(target)
                target_type = fact.get('target_type', 'Location')
                response += f"• **{target}** ({target_type})\n"
                if fact.get('target_desc'):
                    response += f"  {fact['target_desc'][:100]}...\n"
        response += "\n"
    
    response += "---\n\n"
    response += "💡 **Pro Tip:** Combine multiple activities in the same area to save travel time!\n"
    
    return response

def format_fallback_response(query: str, matches: list, graph_facts: list) -> str:
    """
    Format a beautiful response without LLM generation
    Intelligently detects query type and formats accordingly
    """
    query_lower = query.lower()
    
    # Detect if it's an itinerary request
    is_itinerary = any(word in query_lower for word in ['itinerary', 'trip', 'plan', 'days', 'day'])
    is_romantic = 'romantic' in query_lower or 'couple' in query_lower or 'honeymoon' in query_lower
    
    # Extract number of days if mentioned
    days_match = re.search(r'(\d+)\s*day', query_lower)
    num_days = int(days_match.group(1)) if days_match else 3
    
    if is_itinerary:
        return format_itinerary(query, matches, graph_facts, num_days, is_romantic)
    else:
        return format_recommendations(query, matches, graph_facts)

def call_chat(prompt_text, matches, graph_facts):
    """
    Call Gemini Chat API with fallback system
    If Gemini fails/blocks, return formatted fallback
    """
    
    # Try multiple models with different safety settings
    attempts = [
        ("gemini-1.5-flash", "BLOCK_ONLY_HIGH"),
        ("gemini-1.5-flash-latest", "BLOCK_NONE"),
        ("gemini-1.5-pro-latest", "BLOCK_ONLY_HIGH")
    ]
    
    for model_name, block_level in attempts:
        try:
            safety_settings = {
                'HARM_CATEGORY_HARASSMENT': block_level,
                'HARM_CATEGORY_HATE_SPEECH': block_level,
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': block_level,
                'HARM_CATEGORY_DANGEROUS_CONTENT': block_level
            }
            
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt_text,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1500,
                    top_p=0.9
                ),
                safety_settings=safety_settings
            )
            
            # If successful, return the text
            if response.text:
                return response.text
                
        except Exception as e:
            # Continue to next attempt
            continue
    
    # If all Gemini attempts fail, use fallback formatting
    print("ℹ️  Using formatted response (Gemini unavailable)")
    
    # Extract query from prompt (it's at the start)
    query = prompt_text.split('\n')[0].replace("You are a Vietnam travel expert. Answer this query: ", "")
    return format_fallback_response(query, matches, graph_facts)

# -----------------------------
# Interactive Chat
# -----------------------------
def interactive_chat():
    """Main interactive loop"""
    print("=" * 70)
    print("🌟 HYBRID AI TRAVEL ASSISTANT (Powered by Gemini)")
    print("   Pinecone + Neo4j + Gemini AI")
    print("=" * 70)
    print("\n💡 Example queries:")
    print("  • Create a romantic 4 day itinerary for Vietnam")
    print("  • What are the best adventure activities?")
    print("  • Suggest cultural experiences in Hanoi")
    print("  • Plan a budget-friendly weekend trip")
    print("  • What to do in Hanoi?")
    print("  • Best beaches in Central Vietnam")
    print("\nType 'exit' or 'quit' to end the session.\n")
    print("=" * 70 + "\n")
    
    while True:
        query = input("🗣️  Your travel question: ").strip()
        
        if not query:
            continue
            
        if query.lower() in ("exit", "quit", "q"):
            print("\n👋 Thanks for using the Hybrid AI Travel Assistant!")
            break
        
        print("\n" + "=" * 70)
        print("🔄 PROCESSING YOUR REQUEST")
        print("=" * 70 + "\n")
        
        # Step 1: Vector search
        matches = pinecone_query(query, top_k=TOP_K)
        
        if not matches:
            print("⚠️  No matches found in Pinecone. Try rephrasing your query.\n")
            continue
        
        # Step 2: Get node IDs from metadata
        match_ids = []
        for m in matches:
            node_id = m.metadata.get('id') or m.id
            match_ids.append(node_id)
        
        # Step 3: Fetch graph context
        graph_facts = fetch_graph_context(match_ids)
        
        # Step 4: Build prompt
        prompt = build_prompt(query, matches, graph_facts)
        
        # Step 5: Generate response (with automatic fallback)
        print("🤖 Generating response with Gemini AI...\n")
        answer = call_chat(prompt, matches, graph_facts)
        
        # Step 6: Display results
        print("=" * 70)
        print("🤖 ASSISTANT RESPONSE")
        print("=" * 70)
        print(answer)
        print("=" * 70 + "\n")
        
        # Show context summary
        print("📊 Context Used:")
        print(f"  • {len(matches)} semantic matches from Pinecone")
        print(f"  • {len(graph_facts)} graph relationships from Neo4j")
        print("=" * 70 + "\n")

if __name__ == "__main__":
    try:
        interactive_chat()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()
        print("\n✨ Session ended.\n")