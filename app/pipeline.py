import re
import pickle
import numpy as np
from collections import Counter
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

EMOTION_LABELS = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion",
    "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
    "excitement","fear","gratitude","grief","joy","love","nervousness","optimism",
    "pride","realization","relief","remorse","sadness","surprise","neutral"
]

class YouTubeSentimentPipeline:

    def __init__(self, model_path, tokenizer_path):
        self.model = load_model(model_path)
        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)
        self.youtube = build("youtube", "v3", developerKey="AIzaSyCwzrJzwNzfje1xJZmZmBU2L2O-cHWPHk0")

    def _extract_video_id(self, url):
        match = re.search(r"v=([^&]+)", url)
        if not match:
            # Handle potential short URLs or other formats if needed, but basic param is fine for now
            raise ValueError("Invalid YouTube URL")
        return match.group(1)

    def _get_comments(self, video_id, max_comments=20):
        comments = []
        try:
            request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(max_comments, 100), # efficient batching
                order="relevance", # Use relevance for "top" comments
                textFormat="plainText"
            )
            response = request.execute()
            
            while response and len(comments) < max_comments:
                for item in response.get("items", []):
                    text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    comments.append(text)
                    if len(comments) >= max_comments:
                        break
                
                if len(comments) >= max_comments:
                    break
                    
                if "nextPageToken" in response:
                     request = self.youtube.commentThreads().list_next(request, response)
                     response = request.execute()
                else:
                    break

        except HttpError as e:
            # Check if reason is commentsDisabled
            if e.resp.status == 403:
                error_details = e.content.decode('utf-8') if hasattr(e, 'content') else str(e)
                if "commentsDisabled" in error_details:
                     print(f"Comments are disabled for video {video_id}")
                     return None # Signal disabled comments
            print(f"An error occurred fetching comments: {e}")
            return [] # Return empty on other errors to avoid crash
        except Exception as e:
             print(f"Unexpected error: {e}")
             return []

        return comments

    def _predict(self, comments):
        if not comments:
            return []
        sequences = self.tokenizer.texts_to_sequences(comments)
        # Fix: Model was trained with padding='post', leading to 'optimism' bias if 'pre' is used.
        padded = pad_sequences(sequences, maxlen=128, padding='post', truncating='post')
        predictions = self.model.predict(padded)
        emotions = []
        
        for pred in predictions:
            # Logic: All emotions > 0.35. If none, take top 4.
            # pred is a probability distribution over labels
            
            # Get indices and probabilities
            label_probs = list(enumerate(pred))
            # Sort by probability descending
            label_probs.sort(key=lambda x: x[1], reverse=True)
            
            # Filter > 0.35
            high_conf_emotions = [idx for idx, prob in label_probs if prob > 0.35]
            
            if high_conf_emotions:
                for idx in high_conf_emotions:
                    emotions.append(EMOTION_LABELS[idx])
            else:
                # Take top 4
                for idx, _ in label_probs[:4]:
                    emotions.append(EMOTION_LABELS[idx])
                    
        return emotions

    def _generate_review(self, emotion_counts, total):
        if total == 0:
            return "No enough data to generate a review."
            
        top_emotions = [e for e, c in emotion_counts.most_common(5)]
        dominant_emotion = top_emotions[0] if top_emotions else "neutral"
        
        # Categorize emotions for broader context
        pos_set = {"admiration", "amusement", "approval", "caring", "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief"}
        neg_set = {"anger", "annoyance", "disappointment", "disapproval", "disgust", "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"}
        
        pos_count = sum(emotion_counts[e] for e in pos_set if e in emotion_counts)
        neg_count = sum(emotion_counts[e] for e in neg_set if e in emotion_counts)
        
        # Construct dynamic prompt/review
        review = f"Based on the analysis of {total} top comments, here is the audience sentiment summary:\n\n"
        
        if pos_count > neg_count * 1.5:
            review += f"🌟 **Overwhelmingly Positive Reception**: The audience is vibing with this content! The most prominent sentiment is **{dominant_emotion.upper()}**. Viewers are expressing strong appreciation, likely due to the entertaining or helpful nature of the video."
        elif neg_count > pos_count * 1.5:
             review += f"⚠️ **Critical Audience Reaction**: The feedback indicates distinctive dissatisfaction using **{dominant_emotion.upper()}**. Several viewers are expressing concerns or frustration, suggesting the content might be controversial or technical issues were present."
        else:
             review += f"⚖️ **Mixed or Balanced Views**: The audience is split. While some are showing **{dominant_emotion.upper()}**, there's a complex mix of reactions. This often happens with thought-provoking topics or debates."

        review += f"\n\n**Top Emotional Drivers**: {', '.join([e.capitalize() for e in top_emotions[:3]])}."
        
        return review

    def analyze_youtube_video(self, url):
        try:
            video_id = self._extract_video_id(url)
        except ValueError:
            return {"error": "Invalid YouTube URL format"}

        comments = self._get_comments(video_id, max_comments=20)
        
        if comments is None:
            return {"error": "Comments are disabled for this video."}
            
        if not comments:
            return {"error": "No comments found."}

        emotions = self._predict(comments)
        counter = Counter(emotions)
        dominant = counter.most_common(5)
        
        review = self._generate_review(counter, len(comments))

        return {
            "total_comments_analyzed": len(comments),
            "video_review": review,
            "dominant_emotions": dominant,
            "emotion_distribution": dict(counter),
            # Return samples of comments just for UI check if needed, or remove if too large
            "top_comments_sample": comments[:5] 
        }
