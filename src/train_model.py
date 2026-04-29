import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import sys

# Connect to your data processing script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_processing import clean_social_text

# 1. HIGH-INTENSITY DATASET 
# This includes diverse phrasing to make the AI highly sensitive to tone 
high_intensity_data = {
    'positive': [
        "brilliant service!", "Best purchase.", "I am obsessed with this.", 
        "Highly recommend", "Surprisingly high quality.", "Not bad at all, actually great.", 
        "Exceeded my expectations.", "The customer support was top-tier.", "Smooth UI and very fast.", 
        "A true masterpiece.", "Finally, a product that works perfectly!", "I love everything about this.", 
        "Simply incredible experience.", "Total game changer for my workflow.", "So worth the money.", 
        "10/10 would buy again.", "Excellent performance and sleek design.", "Bravo to the team!","very good",
        "nice","excellent", "outstanding", "phenomenal", "exceptional", "perfect", "superb", "good", "great",
        "wonderful", "fantastic", "amazing", "awesome", "efficient", "seamless", "reliable", "intuitive", 
        "helpful", "user-friendly", "love", "delighted", "impressed", "happy", "brilliant", "incredible"
    ],
    'negative': [
        "Absolute garbage product.", "Waste of time and money.", "I hate this.", "Total disaster.","keeps crashing constantly.",
        "Worst experience.", "Horrible support.","Extremely disappointed.", "Don't buy this, it is a scam.", "Terrible quality",
        "feels cheap.", "I want my money back.", "So buggy it is unusable.", "Avoid this at all costs.","A complete letdown.",
        "Nothing works as advertised.", "Poorly designed and incredibly slow.", "I am never using this again.", "Disgusting behavior from staff.", 
        "Utterly useless app.", "terrible", "horrible", "awful", "disastrous", "catastrophic", "worst", "bad", "poor", "disappointing",
        "unsatisfactory", "useless", "lame", "broken", "crashed", "slow", "buggy", "expensive", "confusing", "hate", "angry", "annoyed",
        "frustrated", "regret", "appalling", "not", "never", "none", "cannot", "isnt", "didnt"
    ],
    'neutral': [
        "It is okay, I guess.", "Average experience.", "Does what it says.", "Nothing special.",
        "Fine for the price.", "It is an alright product.", "Neutral feelings about this.", 
        "Standard service, no complaints.", "Just another app.", 
        "It is fine.", "Could be better.", "Ordinary quality.", "Middle of the road.", 
        "It works.", "Decent but a bit expensive."
    ]
}

# Massive expansion to stabilize model weights 
all_text = (high_intensity_data['positive'] * 200) + (high_intensity_data['negative'] * 200) + (high_intensity_data['neutral'] * 200)
all_labels = (['positive'] * (len(high_intensity_data['positive']) * 200)) + \
             (['negative'] * (len(high_intensity_data['negative']) * 200)) + \
             (['neutral'] * (len(high_intensity_data['neutral']) * 200))

df = pd.DataFrame({'text': all_text, 'sentiment': all_labels})

# 2. PREPROCESSING
print("Starting High-Intensity Training on samples...")
df['cleaned_text'] = df['text'].apply(clean_social_text) 

# 3. FEATURE EXTRACTION  
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2)) 
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['sentiment']

# 4. ADVANCED MODEL TRAINING
# 'balanced' weight ensures the AI gives equal attention to all emotions 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42) 
model = LogisticRegression(class_weight='balanced', max_iter=5000)
model.fit(X_train, y_train) 

# 5. METRICS & VISUALIZATION
y_pred = model.predict(X_test) 
print(f"✅ Training Complete! Model Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%") 

# Save Confusion Matrix as your 'Proof of Work' 
if not os.path.exists('outputs'): os.makedirs('outputs')
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('High-Intensity Training: Confusion Matrix')
plt.savefig('outputs/intensity_confusion_matrix.png')

# 6. SAVE ARTIFACTS 
if not os.path.exists('models'): os.makedirs('models')
joblib.dump(model, 'models/sentiment_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
print("🚀 High-Intensity Model and Vectorizer saved successfully!")