# 🚀 Brand Sentinel: Advanced Sentiment Intelligence

**Brand Sentinel** is a high-performance sentiment analysis dashboard designed to track customer emotions in real-time. It uses a hybrid approach, combining **VADER Lexicon** for emotional intensity and **Logistic Regression** for text classification.

## ✨ Key Features
* **Real-time Analysis:** Instant sentiment feedback on social media comments.
* **Batch Processing:** Ability to process 8,000+ rows of CSV data.
* **Semantic UI:** High-contrast color-coded analytics (Green/Red/White).
* **N-Gram Intelligence:** Captures negations like "not good" using Bigram vectorization.

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **ML Logic:** Scikit-learn (Logistic Regression), TF-IDF
- **NLP:** NLTK (VADER)
- **Visuals:** Plotly

## 🚦 How to Run
1. Clone the repo: `git clone https://github.com/princevashist239/Project-1-Machine-learning---Social-Media-Sentiment-Analysis.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the trainer: `python src/train_model.py`
4. Launch the dashboard: `streamlit run app/main.py`
