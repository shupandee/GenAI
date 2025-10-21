"""
Complete System Test for Blue Enigma AI-Hybrid Chat Challenge
Tests all components: Config, Neo4j, Pinecone, Gemini, and Hybrid Chat
"""

import sys
import os
from typing import Dict, List

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print("=" * 70 + "\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.RESET}")


# ============================================
# TEST 1: Configuration
# ============================================
def test_config():
    """Test if config.py is properly set up"""
    print_header("TEST 1: Configuration")
    
    try:
        import config
        print_success("config.py imported successfully")
        
        # Check required attributes
        required_attrs = [
            'PINECONE_API_KEY',
            'GEMINI_API_KEY',
            'NEO4J_URI',
            'NEO4J_USER',
            'NEO4J_PASSWORD',
            'PINECONE_INDEX_NAME',
            'EMBEDDING_DIMENSION',
            'GEMINI_CHAT_MODEL'
        ]
        
        missing = []
        for attr in required_attrs:
            if not hasattr(config, attr):
                missing.append(attr)
            else:
                value = getattr(config, attr)
                if isinstance(value, str) and value.startswith("your-"):
                    print_warning(f"{attr} not configured (still has placeholder)")
                else:
                    print_success(f"{attr} is set")
        
        if missing:
            print_error(f"Missing attributes: {', '.join(missing)}")
            return False
        
        # Validate config
        if hasattr(config, 'validate_config'):
            config.validate_config()
        
        return True
        
    except ImportError as e:
        print_error(f"Cannot import config.py: {e}")
        print_info("Make sure config.py exists in the project directory")
        return False
    except Exception as e:
        print_error(f"Configuration error: {e}")
        return False


# ============================================
# TEST 2: Package Installation
# ============================================
def test_packages():
    """Test if all required packages are installed"""
    print_header("TEST 2: Package Installation")
    
    packages = {
        'neo4j': 'neo4j>=5.14.0',
        'pinecone': 'pinecone-client>=3.0.0',
        'google.generativeai': 'google-generativeai>=0.3.0',
        'pyvis': 'pyvis>=0.3.2',
        'networkx': 'networkx>=3.1',
        'tqdm': 'tqdm',
    }
    
    all_installed = True
    
    for module, package in packages.items():
        try:
            __import__(module)
            print_success(f"{module} is installed")
        except ImportError:
            print_error(f"{module} is NOT installed")
            print_info(f"Install with: pip install {package}")
            all_installed = False
    
    return all_installed


# ============================================
# TEST 3: Neo4j Connection
# ============================================
def test_neo4j():
    """Test Neo4j database connection"""
    print_header("TEST 3: Neo4j Connection")
    
    try:
        from neo4j import GraphDatabase
        import config
        
        print_info(f"Connecting to {config.NEO4J_URI}...")
        
        driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        
        # Test connection
        with driver.session() as session:
            result = session.run("RETURN 'Connected!' AS message")
            message = result.single()["message"]
            print_success(f"Neo4j connected: {message}")
        
        # Check if data is loaded
        with driver.session() as session:
            result = session.run("MATCH (n:Location) RETURN count(n) AS count")
            count = result.single()["count"]
            if count > 0:
                print_success(f"Found {count} Location nodes in database")
            else:
                print_warning("No Location nodes found. Run load_to_neo4j.py first")
        
        driver.close()
        return True
        
    except Exception as e:
        print_error(f"Neo4j connection failed: {e}")
        print_info("Make sure:")
        print_info("  1. Neo4j Desktop is running")
        print_info("  2. Database is started")
        print_info("  3. Password in config.py is correct")
        return False


# ============================================
# TEST 4: Gemini API
# ============================================
def test_gemini():
    """Test Gemini API connection and embedding generation"""
    print_header("TEST 4: Gemini API")
    
    try:
        import google.generativeai as genai
        import config
        
        print_info("Configuring Gemini API...")
        genai.configure(api_key=config.GEMINI_API_KEY)
        print_success("Gemini API configured")
        
        # Test embedding
        print_info("Testing embedding generation...")
        result = genai.embed_content(
            model="models/embedding-001",
            content="Test embedding for Vietnam travel",
            task_type="retrieval_document"
        )
        
        embedding = result['embedding']
        print_success(f"Embedding generated successfully")
        print_info(f"Embedding dimension: {len(embedding)}")
        
        if len(embedding) != config.EMBEDDING_DIMENSION:
            print_warning(f"Expected {config.EMBEDDING_DIMENSION}, got {len(embedding)}")
        
        # Test chat model
        print_info(f"Testing chat model: {config.GEMINI_CHAT_MODEL}...")
        model = genai.GenerativeModel(model_name=config.GEMINI_CHAT_MODEL)
        response = model.generate_content("Say 'Hello' in one word")
        print_success(f"Chat model working: {response.text[:50]}")
        
        return True
        
    except Exception as e:
        print_error(f"Gemini API test failed: {e}")
        print_info("Make sure:")
        print_info("  1. GEMINI_API_KEY is set in config.py")
        print_info("  2. Get key from: https://aistudio.google.com/app/apikey")
        return False


# ============================================
# TEST 5: Pinecone Connection
# ============================================
def test_pinecone():
    """Test Pinecone connection and index"""
    print_header("TEST 5: Pinecone")
    
    try:
        from pinecone import Pinecone
        import config
        
        print_info("Connecting to Pinecone...")
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        print_success("Pinecone connected")
        
        # List indexes
        indexes = [idx.name for idx in pc.list_indexes()]
        print_info(f"Available indexes: {indexes if indexes else 'None'}")
        
        # Check if our index exists
        if config.PINECONE_INDEX_NAME in indexes:
            print_success(f"Index '{config.PINECONE_INDEX_NAME}' exists")
            
            # Get index stats
            index = pc.Index(config.PINECONE_INDEX_NAME)
            stats = index.describe_index_stats()
            print_success(f"Total vectors: {stats.total_vector_count}")
            print_success(f"Dimension: {stats.dimension}")
            
            if stats.total_vector_count == 0:
                print_warning("Index is empty. Run pinecone_upload_gemini.py first")
        else:
            print_warning(f"Index '{config.PINECONE_INDEX_NAME}' not found")
            print_info("Run pinecone_upload_gemini.py to create and populate the index")
        
        return True
        
    except Exception as e:
        print_error(f"Pinecone test failed: {e}")
        print_info("Make sure:")
        print_info("  1. PINECONE_API_KEY is set in config.py")
        print_info("  2. Get key from: https://app.pinecone.io/")
        return False


# ============================================
# TEST 6: Hybrid Retrieval
# ============================================
def test_hybrid_retrieval():
    """Test the complete hybrid retrieval system"""
    print_header("TEST 6: Hybrid Retrieval System")
    
    try:
        import google.generativeai as genai
        from pinecone import Pinecone
        from neo4j import GraphDatabase
        import config
        
        # Initialize components
        print_info("Initializing components...")
        
        genai.configure(api_key=config.GEMINI_API_KEY)
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        index = pc.Index(config.PINECONE_INDEX_NAME)
        driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        
        print_success("All components initialized")
        
        # Test query
        test_query = "romantic places in Vietnam"
        print_info(f"Testing with query: '{test_query}'")
        
        # 1. Generate embedding
        print_info("Generating embedding...")
        result = genai.embed_content(
            model="models/embedding-001",
            content=test_query,
            task_type="retrieval_query"
        )
        embedding = result['embedding']
        print_success(f"Embedding generated ({len(embedding)} dimensions)")
        
        # 2. Query Pinecone
        print_info("Querying Pinecone...")
        results = index.query(
            vector=embedding,
            top_k=3,
            include_metadata=True
        )
        print_success(f"Found {len(results.matches)} matches from Pinecone")
        
        if results.matches:
            print_info("Top match:")
            match = results.matches[0]
            print(f"  - Name: {match.metadata.get('name', 'N/A')}")
            print(f"  - Type: {match.metadata.get('type', 'N/A')}")
            print(f"  - Score: {match.score:.4f}")
        
        # 3. Query Neo4j
        node_ids = [m.id for m in results.matches]
        if node_ids:
            print_info(f"Querying Neo4j for {len(node_ids)} nodes...")
            with driver.session() as session:
                query = """
                MATCH (n:Location)
                WHERE n.id IN $node_ids
                OPTIONAL MATCH (n)-[r]-(m:Location)
                RETURN count(r) AS rel_count
                """
                result = session.run(query, node_ids=node_ids)
                rel_count = result.single()["rel_count"]
                print_success(f"Found {rel_count} relationships in Neo4j")
        
        driver.close()
        print_success("Hybrid retrieval system working correctly!")
        return True
        
    except Exception as e:
        print_error(f"Hybrid retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================
# TEST 7: End-to-End Test
# ============================================
def test_end_to_end():
    """Test complete query flow"""
    print_header("TEST 7: End-to-End Query Test")
    
    try:
        import google.generativeai as genai
        from pinecone import Pinecone
        from neo4j import GraphDatabase
        import config
        
        # Initialize
        genai.configure(api_key=config.GEMINI_API_KEY)
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        index = pc.Index(config.PINECONE_INDEX_NAME)
        driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        model = genai.GenerativeModel(model_name=config.GEMINI_CHAT_MODEL)
        
        # Test query
        query = "What are the top 3 romantic places in Vietnam?"
        print_info(f"Query: {query}")
        
        # Get embedding
        emb_result = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        
        # Search Pinecone
        search_results = index.query(
            vector=emb_result['embedding'],
            top_k=3,
            include_metadata=True
        )
        
        # Build context
        context = "Top matches:\n"
        for i, match in enumerate(search_results.matches, 1):
            meta = match.metadata
            context += f"{i}. {meta.get('name', 'N/A')} - {meta.get('description', 'N/A')[:100]}\n"
        
        # Generate response
        prompt = f"Context:\n{context}\n\nQuery: {query}\n\nProvide a brief answer:"
        response = model.generate_content(prompt)
        
        print_success("Generated Response:")
        print("-" * 70)
        print(response.text)
        print("-" * 70)
        
        driver.close()
        return True
        
    except Exception as e:
        print_error(f"End-to-end test failed: {e}")
        return False


# ============================================
# MAIN TEST RUNNER
# ============================================
def run_all_tests():
    """Run all system tests"""
    print_header("BLUE ENIGMA AI-HYBRID CHAT SYSTEM TEST")
    print(f"{Colors.BOLD}Testing all components...{Colors.RESET}\n")
    
    tests = [
        ("Configuration", test_config),
        ("Package Installation", test_packages),
        ("Neo4j Connection", test_neo4j),
        ("Gemini API", test_gemini),
        ("Pinecone", test_pinecone),
        ("Hybrid Retrieval", test_hybrid_retrieval),
        ("End-to-End", test_end_to_end),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        if result:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")
    
    print("\n" + "=" * 70)
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ ALL TESTS PASSED! ({passed}/{total}){Colors.RESET}")
        print("=" * 70)
        print(f"\n{Colors.GREEN}🚀 You're ready to run: python hybrid_chat.py{Colors.RESET}\n")
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  {passed}/{total} TESTS PASSED{Colors.RESET}")
        print("=" * 70)
        print(f"\n{Colors.YELLOW}Please fix the failing tests before running the system.{Colors.RESET}\n")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Tests interrupted by user{Colors.RESET}")
        sys.exit(1)