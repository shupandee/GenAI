"""
Test script for Communication Scorer
Run this to verify the scorer is working correctly
"""

from scorer import CommunicationScorer
import json

def print_separator():
    print("\n" + "="*80 + "\n")

def main():
    print("Communication Skills Scorer - Test Script")
    print_separator()
    
    # Sample transcript from case study
    sample_transcript = """Hello everyone, myself Muskan, studying in class 8th B section from Christ Public School. 
I am 13 years old. I live with my family. There are 3 people in my family, me, my mother and my father.
One special thing about my family is that they are very kind hearted to everyone and soft spoken. One thing I really enjoy is play, playing cricket and taking wickets.
A fun fact about me is that I see in mirror and talk by myself. One thing people don't know about me is that I once stole a toy from one of my cousin.
My favorite subject is science because it is very interesting. Through science I can explore the whole world and make the discoveries and improve the lives of others. 
Thank you for listening."""
    
    print("Sample Transcript:")
    print("-" * 80)
    print(sample_transcript)
    print_separator()
    
    # Initialize scorer
    print("Initializing scorer...")
    scorer = CommunicationScorer()
    print("✓ Scorer initialized successfully!")
    print_separator()
    
    # Score the transcript
    print("Scoring transcript...")
    result = scorer.score_transcript(sample_transcript)
    print("✓ Scoring complete!")
    print_separator()
    
    # Display results
    print("RESULTS")
    print("="*80)
    print(f"\n🎯 Overall Score: {result['overall_score']}/100")
    print(f"📝 Word Count: {result['words']}")
    print(f"📄 Sentence Count: {result['sentences']}")
    print(f"⏱️  Duration: {result['duration_seconds']} seconds")
    
    print_separator()
    print("DETAILED BREAKDOWN BY CRITERION")
    print("="*80)
    
    for criterion in result['criteria']:
        print(f"\n📊 {criterion['criterion']}")
        print(f"   Score: {criterion['score']}/{criterion['max_score']} (Raw)")
        print(f"   Weighted Score: {criterion['weighted_score']}/{criterion['weight']}")
        print(f"   Weight: {criterion['weight']}%")
        print(f"\n   Details:")
        print_details(criterion['details'], indent=6)
    
    print_separator()
    
    # Save to JSON
    print("Saving results to 'test_results.json'...")
    with open('test_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("✓ Results saved!")
    
    print_separator()
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)

def print_details(details, indent=0):
    """Recursively print nested details"""
    spacing = " " * indent
    
    for key, value in details.items():
        if isinstance(value, dict):
            print(f"{spacing}{key}:")
            print_details(value, indent + 3)
        elif isinstance(value, list):
            print(f"{spacing}{key}: {len(value)} items")
            if len(value) > 0 and not isinstance(value[0], tuple):
                for item in value[:3]:  # Show first 3 items
                    print(f"{spacing}  - {item}")
                if len(value) > 3:
                    print(f"{spacing}  ... and {len(value) - 3} more")
        else:
            print(f"{spacing}{key}: {value}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ ERROR: {str(e)}")
        print("\nPlease ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        raise