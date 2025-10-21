# hybrid_chat_async.py - Async version with parallel processing and intelligent fallback
import os
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import asyncio
import time
import re
from typing import List, Dict
from collections import defaultdict
from functools import lru_cache

import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import config

# -----------------------------
# Configuration
# -----------------------------
EMBED_MODEL = "models/text-embedding-004"
CHAT_MODEL = "gemini-1.5-flash"
TOP_K = 8
INDEX_NAME = config.PINECONE_INDEX_NAME

# -----------------------------
# Initialize Clients
# -----------------------------
genai.configure(api_key=config.GEMINI_API_KEY)
pc = Pinecone(api_key=config.PINECONE_API_KEY)

try:
    # Connect to Pinecone
    if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        print(f"Creating serverless index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=config.EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    index = pc.Index(INDEX_NAME)
    print(f"✓ Connected to Pinecone index: {INDEX_NAME}")
except Exception as e:
    print(f"❌ Error connecting to Pinecone: {e}")
    exit(1)

# Connect to Neo4j
driver = GraphDatabase.driver(
    config.NEO4J_URI, 
    auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)
print(f"✓ Connected to Neo4j at {config.NEO4J_URI}")

# Embedding cache for performance
embedding_cache = {}

# -----------------------------
# Async Helper Functions
# -----------------------------
async def embed_text_async(text: str) -> List[float]:
    """Async embedding generation with caching"""
    if text in embedding_cache:
        return embedding_cache[text]
    
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: genai.embed_content(
                model=EMBED_MODEL,
                content=text,
                task_type="retrieval_query"
            )
        )
        embedding = result['embedding']
        embedding_cache[text] = embedding
        return embedding
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None

async def pinecone_query_async(query_text: str, top_k=TOP_K):
    """Async Pinecone query"""
    vec = await embed_text_async(query_text)
    if vec is None:
        return []
    
    try:
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(
            None,
            lambda: index.query(
                vector=vec,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
        )
        return res.matches
    except Exception as e:
        print(f"❌ Error querying Pinecone: {e}")
        return []

async def fetch_graph_context_async(node_ids: List[str]):
    """Async graph context retrieval with parallel processing"""
    
    async def fetch_node_relationships(nid):
        """Fetch relationships for a single node"""
        loop = asyncio.get_event_loop()
        
        def query_neo4j():
            facts = []
            with driver.session() as session:
                # Updated query to use Entity label
                q = """
                    MATCH (n:Entity {id: $nid})
                    OPTIONAL MATCH (n)-[r]-(m:Entity)
                    RETURN n.name AS source_name,
                           type(r) AS rel, 
                           labels(m) AS labels, 
                           m.id AS id,
                           m.name AS name, 
                           m.type AS type, 
                           m.description AS description,
                           m.region AS region,
                           m.tags AS tags
                    LIMIT 15
                """
                try:
                    recs = session.run(q, nid=nid)
                    for r in recs:
                        if r["rel"]:  # Only add if relationship exists
                            facts.append({
                                "source": nid,
                                "source_name": r["source_name"],
                                "rel": r["rel"],
                                "target_id": r["id"],
                                "target_name": r["name"],
                                "target_type": r["type"],
                                "target_desc": (r["description"] or "")[:300],
                                "target_region": r.get("region", ""),
                                "target_tags": r.get("tags", []),
                                "labels": r["labels"]
                            })
                except Exception as e:
                    pass  # Silent failure for individual nodes
            return facts
        
        return await loop.run_in_executor(None, query_neo4j)
    
    # Fetch all nodes in parallel
    tasks = [fetch_node_relationships(nid) for nid in node_ids[:5]]
    results = await asyncio.gather(*tasks)
    
    # Flatten results
    all_facts = []
    for facts in results:
        all_facts.extend(facts)
    
    return all_facts

async def extract_query_intent_async(user_query: str) -> Dict:
    """Async intent recognition"""
    loop = asyncio.get_event_loop()
    
    def extract_intent():
        intent = {
            "duration": 4,
            "style": "general",
            "locations": [],
            "type": "itinerary"
        }
        
        query_lower = user_query.lower()
        
        # Style detection
        if "romantic" in query_lower or "couple" in query_lower or "honeymoon" in query_lower:
            intent["style"] = "romantic"
        elif "adventure" in query_lower:
            intent["style"] = "adventure"
        elif "cultural" in query_lower or "culture" in query_lower:
            intent["style"] = "cultural"
        elif "beach" in query_lower:
            intent["style"] = "beach"
        elif "budget" in query_lower:
            intent["style"] = "budget"
        
        # Duration extraction
        duration_match = re.search(r'(\d+)\s*day', query_lower)
        if duration_match:
            intent["duration"] = int(duration_match.group(1))
        
        # Location extraction
        cities = ["hanoi", "hue", "hoi an", "da nang", "nha trang", "da lat", 
                  "ho chi minh", "saigon", "mekong", "sapa", "ha long"]
        for city in cities:
            if city in query_lower:
                intent["locations"].append(city.title())
        
        # Type detection
        if any(word in query_lower for word in ['itinerary', 'trip', 'plan', 'days']):
            intent["type"] = "itinerary"
        else:
            intent["type"] = "recommendation"
        
        return intent
    
    return await loop.run_in_executor(None, extract_intent)

# -----------------------------
# Formatting Functions (from sync version)
# -----------------------------
def format_itinerary(query: str, matches: list, graph_facts: list, intent: dict) -> str:
    """Format as a day-by-day itinerary"""
    
    num_days = intent.get('duration', 3)
    is_romantic = intent.get('style') == 'romantic'
    
    response = f"# 🎒 Your {num_days}-Day Vietnam Itinerary\n\n"
    
    if is_romantic:
        response += "💕 *Specially curated for couples*\n\n"
    elif intent.get('style') == 'adventure':
        response += "⛰️ *Adventure-focused itinerary*\n\n"
    elif intent.get('style') == 'cultural':
        response += "🏛️ *Cultural immersion experience*\n\n"
    
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
    
    # Add best time info
    best_times = set()
    for match in matches[:5]:
        bt = match.metadata.get('best_time_to_visit')
        if bt:
            best_times.add(bt)
    
    if best_times:
        response += f"- **Best time to visit:** {', '.join(list(best_times)[:2])}\n"
    
    response += f"- **Estimated activities:** {activities_assigned} unique experiences\n"
    
    # Add style-specific tips
    if is_romantic:
        response += "\n### 💕 Romantic Touches\n"
        response += "- Book sunset activities in advance\n"
        response += "- Consider private tours for intimate experiences\n"
        response += "- Try couples' spa treatments\n"
        response += "- Reserve window seats at restaurants\n"
    elif intent.get('style') == 'adventure':
        response += "\n### ⛰️ Adventure Tips\n"
        response += "- Pack appropriate gear and clothing\n"
        response += "- Book activities in advance during peak season\n"
        response += "- Consider hiring local guides\n"
        response += "- Stay hydrated and bring sun protection\n"
    
    # Add nearby attractions
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
    """Format as a list of recommendations"""
    
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
            
            # Add details
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
            
            # Add tags
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

def format_fallback_response(query: str, matches: list, graph_facts: list, intent: dict) -> str:
    """Format fallback response based on intent"""
    
    if intent.get('type') == 'itinerary':
        return format_itinerary(query, matches, graph_facts, intent)
    else:
        return format_recommendations(query, matches, graph_facts)

# -----------------------------
# Async Chat Generation
# -----------------------------
async def call_chat_async(prompt_text, matches, graph_facts, intent):
    """Async Gemini chat call with fallback"""
    
    # Try multiple models
    attempts = [
        ("gemini-1.5-flash", "BLOCK_ONLY_HIGH"),
        ("gemini-1.5-flash-latest", "BLOCK_NONE"),
        ("gemini-1.5-pro-latest", "BLOCK_ONLY_HIGH")
    ]
    
    loop = asyncio.get_event_loop()
    
    for model_name, block_level in attempts:
        try:
            safety_settings = {
                'HARM_CATEGORY_HARASSMENT': block_level,
                'HARM_CATEGORY_HATE_SPEECH': block_level,
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': block_level,
                'HARM_CATEGORY_DANGEROUS_CONTENT': block_level
            }
            
            model = genai.GenerativeModel(model_name)
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(
                    prompt_text,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=2000,
                        top_p=0.9
                    ),
                    safety_settings=safety_settings
                )
            )
            
            if response.text:
                return response.text
                
        except Exception as e:
            continue
    
    # Fallback to formatted response
    print("ℹ️  Using formatted response (Gemini unavailable)")
    query = prompt_text.split('\n')[0].replace("You are a Vietnam travel expert. Answer this query: ", "")
    return format_fallback_response(query, matches, graph_facts, intent)

def build_prompt(user_query, matches, graph_facts, intent):
    """Build prompt for Gemini"""
    
    vec_context = []
    for i, m in enumerate(matches[:5], 1):
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
    
    graph_context = []
    if graph_facts:
        seen = set()
        for fact in graph_facts[:15]:
            target = fact.get('target_name')
            if target and target not in seen:
                seen.add(target)
                graph_context.append(f"- {target} ({fact.get('target_type', 'N/A')})")
    
    prompt = f"""You are a Vietnam travel expert. Answer this query: {user_query}

Travel Intent: {intent['style']} trip, {intent['duration']} days

Available options:

{chr(10).join(vec_context)}

Related attractions:
{chr(10).join(graph_context[:10]) if graph_context else "No additional connections."}

Provide helpful recommendations with:
1. Specific place names from the list
2. Brief descriptions
3. Practical travel tips
4. Clear, organized format

Keep response positive and concise."""
    
    return prompt

# -----------------------------
# Main Async Processing
# -----------------------------
async def process_query_async(user_query: str):
    """Main async processing pipeline with parallel execution"""
    start_time = time.time()
    
    try:
        # Phase 1: Intent analysis and vector search in parallel
        intent_task = extract_query_intent_async(user_query)
        vector_task = pinecone_query_async(user_query, top_k=TOP_K)
        
        intent, matches = await asyncio.gather(intent_task, vector_task)
        
        if not matches:
            return None, "No results found. Please try a different query."
        
        phase1_time = time.time() - start_time
        
        # Phase 2: Graph context retrieval
        match_ids = [m.metadata.get('id') or m.id for m in matches]
        graph_facts = await fetch_graph_context_async(match_ids)
        
        phase2_time = time.time() - start_time
        
        # Summary
        summary = {
            "cities": len([m for m in matches if m.metadata.get('type') == 'City']),
            "attractions": len([m for m in matches if m.metadata.get('type') == 'Attraction']),
            "hotels": len([m for m in matches if m.metadata.get('type') == 'Hotel']),
            "activities": len([m for m in matches if m.metadata.get('type') == 'Activity']),
            "graph_connections": len(graph_facts)
        }
        
        # Phase 3: Generate response
        prompt = build_prompt(user_query, matches, graph_facts, intent)
        answer = await call_chat_async(prompt, matches, graph_facts, intent)
        
        total_time = time.time() - start_time
        
        return {
            "answer": answer,
            "summary": summary,
            "intent": intent,
            "time": total_time,
            "phase1_time": phase1_time,
            "phase2_time": phase2_time
        }, None
        
    except Exception as e:
        return None, f"Error processing query: {e}"

# -----------------------------
# Interactive Chat Interface
# -----------------------------
async def interactive_chat_async():
    """Async interactive chat interface"""
    print("\n" + "=" * 70)
    print("🌟 HYBRID AI TRAVEL ASSISTANT (ASYNC)")
    print("   Pinecone + Neo4j + Gemini AI")
    print("=" * 70)
    print("\n⚡ Async Mode: Faster parallel processing enabled!")
    print("\n💡 Example queries:")
    print("  • Create a romantic 4 day itinerary for Vietnam")
    print("  • What are the best adventure activities?")
    print("  • Suggest cultural experiences in Hanoi")
    print("  • Best beaches in Central Vietnam")
    print("\nType 'exit' or 'quit' to end")
    print("Type 'help' for more examples")
    print("Type 'stats' for session statistics\n")
    print("=" * 70 + "\n")
    
    query_count = 0
    total_time = 0
    
    while True:
        query = input("🗣️  Your travel question: ").strip()
        
        if not query:
            continue
        
        if query.lower() in ("exit", "quit", "q"):
            if query_count > 0:
                avg_time = total_time / query_count
                print(f"\n📊 Session Statistics:")
                print(f"   Queries processed: {query_count}")
                print(f"   Average response time: {avg_time:.2f}s")
                print(f"   Total time saved: ~{query_count * 0.5:.1f}s (vs sync)")
            print("\n👋 Thanks for using the Travel Assistant!")
            break
        
        if query.lower() == "help":
            print("\n📝 Example queries:")
            print("  • Create a romantic 4 day itinerary for Vietnam")
            print("  • I want to visit Hanoi and Ha Long Bay for 3 days")
            print("  • Recommend beach activities in Nha Trang")
            print("  • Best hotels in Hoi An for a romantic trip")
            print("  • Cultural attractions in Hue")
            print("  • Plan a budget 5 day trip to Northern Vietnam\n")
            continue
        
        if query.lower() == "stats":
            if query_count > 0:
                print(f"\n📊 Current Session:")
                print(f"   Queries: {query_count}")
                print(f"   Avg time: {total_time / query_count:.2f}s")
                print(f"   Cache entries: {len(embedding_cache)}\n")
            else:
                print("\n📊 No queries processed yet\n")
            continue
        
        # Process query
        print("\n" + "=" * 70)
        print("🔄 PROCESSING YOUR REQUEST")
        print("=" * 70 + "\n")
        
        result, error = await process_query_async(query)
        
        if error:
            print(f"❌ {error}\n")
            continue
        
        query_count += 1
        total_time += result['time']
        
        # Display result
        print("=" * 70)
        print("🤖 ASSISTANT RESPONSE")
        print("=" * 70)
        print(result['answer'])
        print("=" * 70 + "\n")
        
        # Display metrics
        print("📊 Context Used:")
        print(f"  • {len(result['summary'])} total items")
        print(f"  • {result['summary']['graph_connections']} graph relationships")
        print(f"  • Intent: {result['intent']['style']} ({result['intent']['duration']} days)")
        print(f"  • Response time: {result['time']:.2f}s")
        print(f"    - Phase 1 (Search): {result['phase1_time']:.2f}s")
        print(f"    - Phase 2 (Graph): {result['phase2_time'] - result['phase1_time']:.2f}s")
        print(f"    - Phase 3 (Generate): {result['time'] - result['phase2_time']:.2f}s")
        print("=" * 70 + "\n")

# -----------------------------
# Main Entry Point
# -----------------------------
def main():
    """Main entry point for async chat"""
    try:
        asyncio.run(interactive_chat_async())
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted. Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}\n")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()

if __name__ == "__main__":
    main()