import re
from collections import Counter
import language_tool_python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
import numpy as np

class CommunicationScorer:
    def __init__(self):
        """Initialize all required models and tools"""
        print("Initializing Communication Scorer...")
        
        # Load sentence transformer for semantic similarity
        print("Loading sentence transformer model...")
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize language tool for grammar checking
        print("Loading grammar checker...")
        self.grammar_tool = language_tool_python.LanguageTool('en-US')
        
        # Initialize sentiment analyzer
        print("Loading sentiment analyzer...")
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Define rubric descriptions for semantic matching
        self.rubric_descriptions = {
            'content_structure': 'personal introduction including name age school family hobbies interests goals achievements',
            'speech_rate': 'speaking pace rhythm tempo speed fluency',
            'language_grammar': 'correct grammar proper sentence structure accurate language usage',
            'clarity': 'clear articulation no filler words concise direct communication',
            'engagement': 'enthusiastic positive confident energetic expressive'
        }
        
        print("Scorer initialized successfully!")
    
    def score_transcript(self, transcript):
        """Main scoring function that combines all approaches"""
        
        # Basic text analysis
        words = transcript.strip().split()
        word_count = len(words)
        sentences = [s.strip() for s in re.split(r'[.!?]+', transcript) if s.strip()]
        sentence_count = len(sentences)
        
        # Calculate duration (assuming sample is 52 seconds as per rubric)
        duration_seconds = 52
        
        # Score each criterion
        content_score = self._score_content_structure(transcript, words, word_count)
        speech_rate_score = self._score_speech_rate(word_count, duration_seconds)
        grammar_score = self._score_language_grammar(transcript, words, word_count)
        clarity_score = self._score_clarity(transcript, words, word_count)
        engagement_score = self._score_engagement(transcript)
        
        # Calculate weighted overall score
        total_score = (
            (content_score['score'] / 30) * 40 +
            (speech_rate_score['score'] / 10) * 10 +
            (grammar_score['score'] / 20) * 20 +
            (clarity_score['score'] / 15) * 15 +
            (engagement_score['score'] / 15) * 15
        )
        
        return {
            'overall_score': round(total_score, 2),
            'words': word_count,
            'sentences': sentence_count,
            'duration_seconds': duration_seconds,
            'criteria': [
                {
                    'criterion': 'Content & Structure',
                    'score': round(content_score['score'], 2),
                    'max_score': 30,
                    'weighted_score': round((content_score['score'] / 30) * 40, 2),
                    'weight': 40,
                    'details': content_score['details']
                },
                {
                    'criterion': 'Speech Rate',
                    'score': round(speech_rate_score['score'], 2),
                    'max_score': 10,
                    'weighted_score': round((speech_rate_score['score'] / 10) * 10, 2),
                    'weight': 10,
                    'details': speech_rate_score['details']
                },
                {
                    'criterion': 'Language & Grammar',
                    'score': round(grammar_score['score'], 2),
                    'max_score': 20,
                    'weighted_score': round((grammar_score['score'] / 20) * 20, 2),
                    'weight': 20,
                    'details': grammar_score['details']
                },
                {
                    'criterion': 'Clarity',
                    'score': round(clarity_score['score'], 2),
                    'max_score': 15,
                    'weighted_score': round((clarity_score['score'] / 15) * 15, 2),
                    'weight': 15,
                    'details': clarity_score['details']
                },
                {
                    'criterion': 'Engagement',
                    'score': round(engagement_score['score'], 2),
                    'max_score': 15,
                    'weighted_score': round((engagement_score['score'] / 15) * 15, 2),
                    'weight': 15,
                    'details': engagement_score['details']
                }
            ]
        }
    
    def _score_content_structure(self, transcript, words, word_count):
        """Score content and structure (40% weight)"""
        total_score = 0
        details = {}
        
        # 1. Salutation Level (5 points)
        salutation_keywords = {
            'excellent': ['i am excited to introduce', 'feeling great'],
            'good': ['good morning', 'good afternoon', 'good evening', 'good day', 'hello everyone'],
            'normal': ['hi', 'hello']
        }
        
        text_lower = transcript.lower()
        salutation_score = 0
        salutation_found = "No salutation"
        
        for phrase in salutation_keywords['excellent']:
            if phrase in text_lower:
                salutation_score = 5
                salutation_found = "Excellent"
                break
        
        if salutation_score == 0:
            for phrase in salutation_keywords['good']:
                if phrase in text_lower:
                    salutation_score = 4
                    salutation_found = "Good"
                    break
        
        if salutation_score == 0:
            for phrase in salutation_keywords['normal']:
                if phrase in text_lower:
                    salutation_score = 2
                    salutation_found = "Normal"
                    break
        
        total_score += salutation_score
        details['salutation'] = {
            'score': salutation_score,
            'max_score': 5,
            'level': salutation_found,
            'keywords_found': [kw for kw in salutation_keywords['good'] + salutation_keywords['normal'] if kw in text_lower]
        }
        
        # 2. Key Word Presence (20 points)
        required_keywords = {
            'name': ['name', 'myself', 'i am', "i'm"],
            'age': ['age', 'years old', 'year old', 'class'],
            'school': ['school', 'college', 'university', 'studying'],
            'family': ['family', 'mother', 'father', 'parents', 'brother', 'sister', 'people in my family'],
            'hobbies': ['hobby', 'hobbies', 'enjoy', 'like', 'love', 'interest', 'play'],
            'unique': ['special', 'unique', 'fun fact', 'about me']
        }
        
        optional_keywords = {
            'origin': ['from', 'live in', 'born in'],
            'ambition': ['goal', 'dream', 'want to', 'aspire', 'future'],
            'achievement': ['achievement', 'award', 'won', 'accomplished']
        }
        
        found_required = []
        for category, keywords in required_keywords.items():
            if any(kw in text_lower for kw in keywords):
                found_required.append(category)
        
        found_optional = []
        for category, keywords in optional_keywords.items():
            if any(kw in text_lower for kw in keywords):
                found_optional.append(category)
        
        # Must have all 6 required elements for full 20 points
        keyword_score = len(found_required) * (20 / 6)
        
        # Optional elements give bonus (max 2 points bonus)
        if len(found_required) == 6:
            bonus = min(2, len(found_optional))
            keyword_score = min(20, keyword_score + bonus)
        
        total_score += keyword_score
        details['keywords'] = {
            'score': round(keyword_score, 2),
            'max_score': 20,
            'required_found': found_required,
            'optional_found': found_optional,
            'required_count': len(found_required),
            'total_required': 6
        }
        
        # 3. Flow/Structure (5 points)
        has_greeting = any(word in text_lower for word in ['hello', 'hi', 'good morning', 'good afternoon', 'good evening'])
        has_name = any(word in text_lower for word in ['myself', 'my name', 'i am', "i'm"])
        has_details = len(found_required) >= 3
        has_closing = any(word in text_lower for word in ['thank', 'thanks', 'grateful'])
        
        flow_score = 0
        if has_greeting:
            flow_score += 1.25
        if has_name:
            flow_score += 1.25
        if has_details:
            flow_score += 1.25
        if has_closing:
            flow_score += 1.25
        
        total_score += flow_score
        details['flow'] = {
            'score': round(flow_score, 2),
            'max_score': 5,
            'has_greeting': has_greeting,
            'has_name': has_name,
            'has_details': has_details,
            'has_closing': has_closing
        }
        
        # Semantic similarity with rubric description
        transcript_embedding = self.semantic_model.encode(transcript)
        rubric_embedding = self.semantic_model.encode(self.rubric_descriptions['content_structure'])
        similarity = np.dot(transcript_embedding, rubric_embedding) / (
            np.linalg.norm(transcript_embedding) * np.linalg.norm(rubric_embedding)
        )
        
        details['semantic_similarity'] = {
            'score': round(float(similarity), 3),
            'interpretation': 'High' if similarity > 0.5 else 'Medium' if similarity > 0.3 else 'Low'
        }
        
        return {
            'score': total_score,
            'details': details
        }
    
    def _score_speech_rate(self, word_count, duration_seconds):
        """Score speech rate (10% weight)"""
        wpm = (word_count / duration_seconds) * 60
        
        if 111 <= wpm <= 140:
            score = 10
            rating = "Ideal"
        elif 81 <= wpm <= 110:
            score = 6
            rating = "Slow"
        elif 141 <= wpm <= 160:
            score = 6
            rating = "Fast"
        elif wpm > 161:
            score = 2
            rating = "Too Fast"
        else:
            score = 2
            rating = "Too Slow"
        
        return {
            'score': score,
            'details': {
                'wpm': round(wpm, 1),
                'rating': rating,
                'feedback': self._get_speech_rate_feedback(wpm)
            }
        }
    
    def _get_speech_rate_feedback(self, wpm):
        if 111 <= wpm <= 140:
            return "Perfect speaking pace. Clear and easy to follow."
        elif 81 <= wpm <= 110:
            return "Speaking a bit slow. Try to increase your pace slightly."
        elif 141 <= wpm <= 160:
            return "Speaking a bit fast. Slow down for better clarity."
        elif wpm > 161:
            return "Too fast! Slow down significantly for better comprehension."
        else:
            return "Too slow. Increase your pace to maintain engagement."
    
    def _score_language_grammar(self, transcript, words, word_count):
        """Score language and grammar (20% weight)"""
        
        # 1. Grammar errors (10 points)
        matches = self.grammar_tool.check(transcript)
        grammar_errors = len(matches)
        errors_per_100 = (grammar_errors / word_count) * 100 if word_count > 0 else 0
        
        if errors_per_100 < 0.3:
            grammar_score = 10
        elif errors_per_100 < 0.5:
            grammar_score = 8
        elif errors_per_100 < 0.7:
            grammar_score = 6
        elif errors_per_100 < 0.9:
            grammar_score = 4
        else:
            grammar_score = 2
        
        # 2. Vocabulary richness - TTR (10 points)
        unique_words = set(word.lower() for word in words)
        ttr = len(unique_words) / len(words) if words else 0
        
        if 0.7 <= ttr <= 0.89:
            vocab_score = 8
        elif 0.5 <= ttr <= 0.69:
            vocab_score = 6
        elif 0.3 <= ttr <= 0.49:
            vocab_score = 4
        elif ttr >= 0.9:
            vocab_score = 10
        else:
            vocab_score = 2
        
        total_score = grammar_score + vocab_score
        
        return {
            'score': total_score,
            'details': {
                'grammar': {
                    'score': grammar_score,
                    'errors_count': grammar_errors,
                    'errors_per_100': round(errors_per_100, 2),
                    'sample_errors': [match.ruleId for match in matches[:3]]
                },
                'vocabulary': {
                    'score': vocab_score,
                    'ttr': round(ttr, 3),
                    'unique_words': len(unique_words),
                    'total_words': len(words)
                }
            }
        }
    
    def _score_clarity(self, transcript, words, word_count):
        """Score clarity - filler word rate (15% weight)"""
        
        filler_words = ['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 
                        'right', 'i mean', 'well', 'kinda', 'sort of', 'okay', 'hmm', 'ah']
        
        text_lower = transcript.lower()
        filler_count = 0
        found_fillers = []
        
        for filler in filler_words:
            pattern = r'\b' + re.escape(filler) + r'\b'
            matches = re.findall(pattern, text_lower)
            if matches:
                filler_count += len(matches)
                found_fillers.append((filler, len(matches)))
        
        filler_rate = (filler_count / word_count) * 100 if word_count > 0 else 0
        
        if filler_rate == 0:
            score = 15
        elif filler_rate <= 3:
            score = 15
        elif filler_rate <= 6:
            score = 12
        elif filler_rate <= 9:
            score = 9
        elif filler_rate <= 12:
            score = 6
        else:
            score = 3
        
        return {
            'score': score,
            'details': {
                'filler_rate': round(filler_rate, 2),
                'filler_count': filler_count,
                'found_fillers': found_fillers,
                'feedback': self._get_clarity_feedback(filler_rate)
            }
        }
    
    def _get_clarity_feedback(self, filler_rate):
        if filler_rate <= 3:
            return "Excellent clarity! Very few filler words."
        elif filler_rate <= 6:
            return "Good clarity with minimal filler words."
        elif filler_rate <= 9:
            return "Moderate filler word usage. Try to reduce them."
        elif filler_rate <= 12:
            return "High filler word usage. Practice reducing fillers."
        else:
            return "Very high filler word usage. Focus on eliminating fillers."
    
    def _score_engagement(self, transcript):
        """Score engagement using sentiment analysis (15% weight)"""
        
        # VADER sentiment analysis
        sentiment_scores = self.sentiment_analyzer.polarity_scores(transcript)
        compound_score = sentiment_scores['compound']
        
        # Scoring based on positivity
        if compound_score >= 0.9:
            score = 15
        elif 0.7 <= compound_score < 0.9:
            score = 12
        elif 0.5 <= compound_score < 0.7:
            score = 9
        elif 0.3 <= compound_score < 0.5:
            score = 6
        else:
            score = 3
        
        # Additional check for enthusiastic words
        enthusiastic_words = ['excited', 'enthusiastic', 'confident', 'grateful', 
                              'love', 'enjoy', 'passionate', 'great', 'wonderful', 
                              'amazing', 'fantastic', 'interesting']
        
        text_lower = transcript.lower()
        enthusiastic_count = sum(1 for word in enthusiastic_words if word in text_lower)
        
        return {
            'score': score,
            'details': {
                'sentiment_score': round(compound_score, 3),
                'positive': round(sentiment_scores['pos'], 3),
                'negative': round(sentiment_scores['neg'], 3),
                'neutral': round(sentiment_scores['neu'], 3),
                'enthusiastic_words_count': enthusiastic_count,
                'interpretation': self._get_engagement_feedback(compound_score)
            }
        }
    
    def _get_engagement_feedback(self, compound_score):
        if compound_score >= 0.9:
            return "Highly engaging and enthusiastic!"
        elif compound_score >= 0.7:
            return "Very positive and engaging."
        elif compound_score >= 0.5:
            return "Moderately positive."
        elif compound_score >= 0.3:
            return "Slightly positive but could be more engaging."
        else:
            return "Low engagement. Try to add more enthusiasm and positive expressions."