import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sys
import os

# Ensure the app can find your processing scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_processing import clean_social_text, get_sentiment_intensity

# Load ML artifacts 
try:
    model = joblib.load('models/sentiment_model.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
except:
    st.error("Model files not found! Please run 'python src/train_model.py' first.")

st.set_page_config(page_title="Brand Sentinel AI", layout="wide")
st.title("🚀 Brand Sentinel: Advanced Sentiment Intelligence")
st.markdown("### Real-time Customer Emotion Tracking for Industry Use Cases")

# Sidebar navigation 
menu = ["Real-time Analysis", "Batch Insights", "Project Documentation"]
choice = st.sidebar.selectbox("Navigate", menu)

# --- SECTION 1: REAL-TIME ANALYSIS ---
if choice == "Real-time Analysis":
    col1, col2 = st.columns(2)
    with col1:
        user_input = st.text_area("Enter Social Media Comment:")
        
        if st.button("Analyze"):
            if user_input:
                # 1. Process the input
                scores = get_sentiment_intensity(user_input)
                compound = scores['compound']

                # 2. Threshold Logic
                if compound >= 0.05:
                    final_label = "POSITIVE"
                    color = "green"
                elif compound <= -0.05:
                    final_label = "NEGATIVE"
                    color = "red"
                else:
                    final_label = "NEUTRAL"
                    color = "white"

                # 3. Display Results
                st.markdown(f"### Result: <span style='color:{color}'>{final_label}</span>", unsafe_allow_html=True)
                st.write("#### 📊 Emotional Breakdown (VADER):")
                st.json(scores)
            else:
                st.warning("Please enter text to analyze.")

# --- SECTION 2: BATCH INSIGHTS (CSV UPLOAD) ---
elif choice == "Batch Insights":
    st.subheader("Upload Social Media Data")
    uploaded_file = st.file_uploader("Upload CSV (Supported Columns: 'review' or 'text')", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # 1. Dynamic Column Identification
        target_col = 'review' if 'review' in df.columns else 'text' if 'text' in df.columns else None
        
        if target_col:
            st.write(f"### Data Preview (Detected Column: '{target_col}')")
            st.dataframe(df) # Shows the full scrollable dataset
            
            if st.button("Run Batch Analysis"):
                with st.spinner('Analyzing full dataset...'):
                    # 2. Vectorized AI Inference
                    df['cleaned'] = df[target_col].apply(clean_social_text)
                    df['sentiment'] = model.predict(vectorizer.transform(df['cleaned']))
                    
                st.success(f"Analysis Complete! Processed {len(df)} rows.")
                
                # 3. Comprehensive Results Table
                st.write("### Complete Analysis Table")
                st.dataframe(df[[target_col, 'sentiment']]) 
                
                st.divider()

                # 4. Strategic Visual Analytics (INDENTED INSIDE THE BUTTON)
                col_pie, col_stats = st.columns(2)
                
                with col_pie:
                    st.write("### 📊 Sentiment Distribution")
                    fig = px.pie(df, names='sentiment', hole=0.4, 
                                 color_discrete_map={
                                     'positive': '#238636', 
                                     'negative': '#da3633', 
                                     'neutral': '#8b949e'
                                 })
                    fig.update_layout(margin=dict(t=20, b=20, l=20, r=20))
                    st.plotly_chart(fig, use_container_width=True)

                with col_stats:
                    st.write("### 📈 Key Performance Indicators")
                    
                    total = len(df)
                    pos_count = (df['sentiment'] == 'positive').sum()
                    neg_count = (df['sentiment'] == 'negative').sum()
                    pos_ratio = pos_count / total
                    neg_ratio = neg_count / total
                    
                    # Calculated Metric Cards
                    st.metric(
                        label="Brand Approval Rating", 
                        value=f"{pos_ratio:.1%}", 
                        delta=f"{(pos_ratio - neg_ratio):.1%} Net Score"
                    )
                    st.metric(
                        label="Critical Action Items", 
                        value=neg_count, 
                        delta="Negative Feedback", 
                        delta_color="inverse"
                    )
                    st.metric(
                        label="Market Neutrality", 
                        value=f"{(df['sentiment'] == 'neutral').mean():.1%}"
                    )
        else:
            st.error("Schema Error: Please ensure your CSV contains a column named 'review' or 'text'.")
# --- SECTION 3: DOCUMENTATION ---
elif choice == "Project Documentation":
    st.header("📄 Project Methodology & Architecture")
    
    st.write("""
    **Brand Sentinel** is an end-to-end Machine Learning pipeline designed to convert unstructured 
    social media noise into actionable brand intelligence.
    """)

    # --- THE TECH STACK PIPELINE ---
    st.subheader("🚀 The Intelligence Pipeline")
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.success("##### 📥 1. Ingestion")
        st.caption("Supports massive CSV batch processing (8000+ rows) and real-time text streams.")
        
    with c2:
        st.success("##### 🧪 2. NLP Prep")
        st.caption("NLTK-powered normalization, stop-word removal, and case folding.")

    with c3:
        st.success("##### 🧠 3. ML Logic")
        st.caption("TF-IDF Vectorization with Bigram context (ngram_range 1,2) + Logistic Regression.")

    with c4:
        st.success("##### 📊 4. Analytics")
        st.caption("Semantic color-mapped Plotly visuals and real-time KPI delta tracking.")

    st.divider()

    # --- BUSINESS USE CASE ---
    st.subheader("💼 Industry Use Cases")
    st.write("""
    This dashboard simulates the reputation management systems used by Tier-1 brands like **Zomato** and **Netflix** to:
    * **Detect Brand Crisis:** Instant alerts for spikes in 'Negative' feedback.
    * **Monitor Market Neutrality:** Tracking passive customer sentiment to drive engagement.
    * **Strategic KPI Tracking:** Calculating Brand Approval Ratings and Net Sentiment Scores.
    """)
    
    st.info("💡 **Tech Tip:** This project utilizes a hybrid sentiment engine, combining VADER Lexicon-based emotional intensity with custom-trained Logistic Regression classification.")