"""
Sample API Client for Communication Scorer
Demonstrates how to interact with the API
"""

import requests
import json
import sys

class ScorerAPIClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        
    def health_check(self):
        """Check if API is healthy"""
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Health check failed: {e}")
            return None
    
    def score_transcript(self, transcript):
        """Score a transcript"""
        try:
            response = requests.post(
                f"{self.base_url}/api/score",
                json={"transcript": transcript},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Scoring failed: {e}")
            if hasattr(e.response, 'text'):
                print(f"Error details: {e.response.text}")
            return None

def print_results(result):
    """Pretty print scoring results"""
    if not result:
        print("No results to display")
        return
    
    print("\n" + "="*80)
    print("SCORING RESULTS")
    print("="*80)
    
    print(f"\n🎯 Overall Score: {result['overall_score']}/100")
    print(f"📝 Words: {result['words']}")
    print(f"📄 Sentences: {result['sentences']}")
    
    print("\n" + "-"*80)
    print("CRITERION BREAKDOWN")
    print("-"*80)
    
    for criterion in result['criteria']:
        print(f"\n{criterion['criterion']}")
        print(f"  Raw Score: {criterion['score']}/{criterion['max_score']}")
        print(f"  Weighted: {criterion['weighted_score']}/{criterion['weight']}")
        
        # Print key details
        if 'details' in criterion:
            if 'feedback' in criterion['details']:
                print(f"  Feedback: {criterion['details']['feedback']}")
            elif 'salutation' in criterion['details']:
                print(f"  Salutation: {criterion['details']['salutation']['level']}")
            elif 'wpm' in criterion['details']:
                print(f"  WPM: {criterion['details']['wpm']} ({criterion['details']['rating']})")
    
    print("\n" + "="*80 + "\n")

def main():
    # Sample transcript
    sample_transcript = """Hello everyone, myself Muskan, studying in class 8th B section from Christ Public School. 
I am 13 years old. I live with my family. There are 3 people in my family, me, my mother and my father.
One special thing about my family is that they are very kind hearted to everyone and soft spoken. 
One thing I really enjoy is play, playing cricket and taking wickets.
A fun fact about me is that I see in mirror and talk by myself. 
One thing people don't know about me is that I once stole a toy from one of my cousin.
My favorite subject is science because it is very interesting. 
Through science I can explore the whole world and make the discoveries and improve the lives of others. 
Thank you for listening."""
    
    # Get API URL from command line or use default
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    
    print(f"Connecting to API at: {api_url}")
    client = ScorerAPIClient(api_url)
    
    # Health check
    print("\nPerforming health check...")
    health = client.health_check()
    if health:
        print(f"✓ API is healthy: {health}")
    else:
        print("✗ API health check failed!")
        return
    
    # Score transcript
    print("\nScoring sample transcript...")
    print("-"*80)
    print(sample_transcript[:200] + "...")
    print("-"*80)
    
    result = client.score_transcript(sample_transcript)
    
    if result:
        print_results(result)
        
        # Save to file
        with open('api_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        print("Results saved to 'api_results.json'")
    else:
        print("✗ Scoring failed!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        raise