import streamlit as st
import requests
import pandas as pd

# Page configuration for a more 'premium' feel
st.set_page_config(page_title="YouTube Sentiment Insights", page_icon="📊", layout="wide")

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF0000;
        color: white;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    .insight-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎥 YouTube Emotion & Insight Analysis")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    url = st.text_input("Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
    analyze_button = st.button("Analyze Video Sentiment")

if analyze_button:
    if not url:
        st.warning("Please enter a valid YouTube URL first.")
    else:
        with st.spinner("Analyzing comments... This may take a few seconds."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/analyze",
                    json={"youtube_url": url}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if "error" in data:
                        st.error(f"❌ {data['error']}")
                    else:
                        st.success("Analysis Complete!")
                        
                        # Display Insights
                        st.subheader("📝 Video Review & Insights")
                        st.info(data.get("video_review", "No review generated."))
                        
                        m_col1, m_col2, m_col3 = st.columns(3)
                        m_col1.metric("Comments Analyzed", data.get("total_comments_analyzed", 0))
                        
                        # Dominant Emotions
                        st.subheader("📊 Emotion Distribution")
                        # Prepare data for bar chart
                        dist = data.get("emotion_distribution", {})
                        if dist:
                            chart_data = pd.DataFrame(dist.items(), columns=['Emotion', 'Count']).sort_values(by='Count', ascending=False)
                            st.bar_chart(chart_data.set_index('Emotion'))
                        
                        # Top Comments Sample
                        with st.expander("💬 View Top Comments Sample"):
                            for i, comment in enumerate(data.get("top_comments_sample", []), 1):
                                st.markdown(f"**{i}.** {comment}")
                                
                else:
                    st.error(f"Server Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Connection Error: Could not reach the API. Make sure the FastAPI server is running. ({e})")

with col2:
    st.markdown("""
    ### How it works
    1. Paste a YouTube link.
    2. We fetch the top **relevant comments**.
    3. Our **Sentiment AI** predicts emotions like Admiration, Joy, or Disappointment.
    4. We generate a **Real-World Insight** report for you.
    """)
    if "data" in locals() and "error" not in data:
         st.write("### Dominant Emotions")
         for emotion, count in data.get("dominant_emotions", []):
             st.write(f"- **{emotion.capitalize()}**: {count}")
