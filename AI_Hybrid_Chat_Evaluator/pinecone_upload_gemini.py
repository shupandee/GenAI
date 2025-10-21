# upload_to_pinecone.py
"""
Pinecone Upload Script with Gemini Embeddings
Updated to match Neo4j schema
"""

import json
import time
from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from tqdm import tqdm
from config import (
    PINECONE_API_KEY,
    GEMINI_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_DIMENSION
)

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Use the same embedding model for consistency
EMBEDDING_MODEL = "models/text-embedding-004"

def get_gemini_embedding(text: str) -> List[float]:
    """
    Generate embedding using Gemini's embedding model
    IMPORTANT: Using text-embedding-004 for consistency with queries
    """
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return [0.0] * EMBEDDING_DIMENSION


def create_pinecone_index():
    """Create Pinecone index with proper configuration"""
    index_name = PINECONE_INDEX_NAME
    
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name in existing_indexes:
        print(f"⚠️  Index '{index_name}' already exists.")
        response = input("Delete and recreate? (y/n): ").strip().lower()
        if response == 'y':
            print(f"Deleting index '{index_name}'...")
            pc.delete_index(index_name)
            time.sleep(5)
        else:
            print("Using existing index...")
            return pc.Index(index_name)
    
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    
    print("Waiting for index to be ready...")
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
    
    print(f"✅ Index '{index_name}' created successfully!")
    return pc.Index(index_name)


def load_travel_data(filepath: str = "vietnam_travel_dataset.json") -> List[Dict]:
    """Load travel dataset from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} travel nodes from {filepath}")
        return data
    except FileNotFoundError:
        print(f"❌ Error: File '{filepath}' not found!")
        raise
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in '{filepath}'!")
        raise


def prepare_and_upload_vectors(index, travel_data: List[Dict], batch_size: int = 50):
    """
    Generate embeddings and upload to Pinecone in batches
    Updated to include all relevant metadata fields
    """
    total_nodes = len(travel_data)
    print(f"\n🚀 Processing {total_nodes} nodes...")
    print(f"⏳ Generating embeddings with Gemini {EMBEDDING_MODEL}...\n")
    
    vectors_batch = []
    successful = 0
    failed = 0
    
    for idx, node in enumerate(tqdm(travel_data, desc="Processing nodes")):
        try:
            # Create rich text representation for embedding
            text_parts = [
                node.get('name', ''),
                node.get('description', ''),
                node.get('type', ''),
                node.get('region', ''),
                ' '.join(node.get('tags', [])),
                node.get('semantic_text', '')
            ]
            text_content = ' '.join(filter(None, text_parts))
            
            # Generate Gemini embedding
            embedding = get_gemini_embedding(text_content)
            
            if not embedding or len(embedding) != EMBEDDING_DIMENSION:
                print(f"\n⚠️  Skipping node {idx}: Invalid embedding")
                failed += 1
                continue
            
            # Prepare comprehensive metadata (Pinecone limits: strings up to 40KB)
            metadata = {
                "id": str(node.get("id", f"node_{idx}")),
                "name": str(node.get("name", ""))[:500],
                "type": str(node.get("type", ""))[:100],
                "description": str(node.get("description", ""))[:1000],
                "region": str(node.get("region", ""))[:200],
                "location": str(node.get("location", ""))[:200],
                "cost": str(node.get("cost", ""))[:100],
                "duration": str(node.get("duration", ""))[:100],
                "best_time_to_visit": str(node.get("best_time_to_visit", ""))[:100],
                "tags": ','.join(node.get("tags", []))[:500],  # Store as comma-separated
            }
            
            # Remove empty values
            metadata = {k: v for k, v in metadata.items() if v and v != ""}
            
            # Create vector dictionary
            vector_id = str(node.get("id", f"node_{idx}"))
            vector = {
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            }
            vectors_batch.append(vector)
            
            # Upload batch when size is reached
            if len(vectors_batch) >= batch_size:
                try:
                    index.upsert(vectors=vectors_batch)
                    successful += len(vectors_batch)
                    vectors_batch = []
                except Exception as e:
                    print(f"\n❌ Error uploading batch: {e}")
                    failed += len(vectors_batch)
                    vectors_batch = []
            
            # Rate limiting for Gemini API
            if (idx + 1) % 10 == 0:
                time.sleep(0.5)
                
        except Exception as e:
            print(f"\n⚠️  Error processing node {idx}: {e}")
            failed += 1
            continue
    
    # Upload remaining vectors
    if vectors_batch:
        try:
            index.upsert(vectors=vectors_batch)
            successful += len(vectors_batch)
        except Exception as e:
            print(f"\n❌ Error uploading final batch: {e}")
            failed += len(vectors_batch)
    
    print(f"\n✨ Upload complete!")
    print(f"   ✅ Successfully uploaded: {successful}")
    print(f"   ❌ Failed: {failed}")
    
    # Print index statistics
    time.sleep(2)
    try:
        stats = index.describe_index_stats()
        print(f"\n📊 Index Statistics:")
        print(f"   Total vectors: {stats.total_vector_count}")
        print(f"   Dimension: {stats.dimension}")
    except Exception as e:
        print(f"⚠️  Could not fetch index stats: {e}")


def test_query(index):
    """Test the index with sample queries"""
    print("\n🔍 Testing with sample queries...\n")
    
    test_queries = [
        "romantic places to visit in Vietnam",
        "cultural experiences in Hanoi",
        "adventure activities in Northern Vietnam"
    ]
    
    for query_text in test_queries:
        try:
            print(f"Query: '{query_text}'")
            
            # Use same model for query as for indexing
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=query_text,
                task_type="retrieval_query"  # Note: query task type
            )
            query_embedding = result['embedding']
            
            results = index.query(
                vector=query_embedding,
                top_k=3,
                include_metadata=True
            )
            
            if results and results.matches:
                for i, match in enumerate(results.matches[:3], 1):
                    score = match.score
                    metadata = match.metadata
                    name = metadata.get('name', 'N/A')
                    node_type = metadata.get('type', 'N/A')
                    region = metadata.get('region', 'N/A')
                    
                    print(f"  {i}. {name} ({node_type}) - {region}")
                    print(f"     Score: {score:.4f}, ID: {match.id}")
            print()
                
        except Exception as e:
            print(f"  ❌ Error during test query: {e}\n")


def main():
    """Main execution function"""
    print("=" * 70)
    print("🌟 PINECONE UPLOAD WITH GEMINI EMBEDDINGS")
    print("   Vietnam Travel RAG System")
    print("=" * 70)
    
    try:
        # Step 1: Create/Connect to Pinecone index
        print("\n📌 Step 1: Setting up Pinecone index...")
        index = create_pinecone_index()
        
        # Step 2: Load travel data
        print("\n📌 Step 2: Loading travel data...")
        travel_data = load_travel_data()
        
        # Step 3: Generate embeddings and upload
        print("\n📌 Step 3: Generating embeddings and uploading to Pinecone...")
        prepare_and_upload_vectors(index, travel_data)
        
        # Step 4: Test queries
        print("\n📌 Step 4: Running test queries...")
        test_query(index)
        
        print("\n" + "=" * 70)
        print("✨ SUCCESS! Pinecone upload complete!")
        print("=" * 70)
        print("\n📋 Next Steps:")
        print("   1. ✅ Verify at: https://app.pinecone.io/")
        print("   2. 🚀 Run: python query_rag.py")
        print("   3. 💬 Ask about Vietnam travel!")
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()