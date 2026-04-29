import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize NLP tools
try:
    stop_words = set(stopwords.words('english'))
    sid = SentimentIntensityAnalyzer()
except:
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))
    sid = SentimentIntensityAnalyzer()

def clean_social_text(text):
    """Cleans text and removes social media noise."""
    text = str(text).lower()
    text = re.sub(r"http\S+|[^A-Za-z\s]", "", text) # Remove URLs and special characters
    stop_words = set(stopwords.words('english'))
    return " ".join([word for word in text.split() if word not in stop_words])

def get_sentiment_intensity(text):
    """Returns granular scores: Negative, Neutral, Positive, and Compound."""
    return sid.polarity_scores(text)

if __name__ == "__main__":
    sample = "This new Zomato update is AWESOME! 🚀"
    print(f"Cleaned: {clean_social_text(sample)}")
    print(f"Intensity: {get_sentiment_intensity(sample)}")