import streamlit as st
import requests
import os

# Page Configuration
st.set_page_config(
    page_title="Amazon Review Analyzer",
    page_icon="🛒",
    layout="wide"
)

# Header & Styling
st.title("🛒 Amazon Review Sentiment Analysis")
st.markdown("""
    <style>
    .big-font { font-size:18px !important; }
    </style>
    <p class="big-font">
        Enter the review details below to predict user sentiment using our MLOps Inference API.
    </p>
""", unsafe_allow_html=True)

st.divider()

# Inputs Section
st.subheader("📝 New Review")

col1, col2 = st.columns([1, 2])

with col1:
    # Input for the Review Title
    review_title = st.text_input(
        "Review Title",
        placeholder="e.g., Great product but..."
    )

with col2:
    # Input for the Review Content
    review_content = st.text_area(
        "Review Content",
        placeholder="e.g., I bought this item last week and I am very satisfied with the quality...",
        height=100
    )

# Logic & API Call
if st.button("Analyze Sentiment", type="primary"):
    
    if not review_title or not review_content:
        st.warning("Please fill in both the Title and the Content.")
    else:
        # Concatenation (Title + Content)
        combined_text = f"{review_title} {review_content}"
        
        # API Configuration
        # Uses an environment variable for flexibility (local vs docker vs aws)
        API_URL = os.getenv("API_URL")

        # Visual feedback while waiting for response
        with st.spinner("Analyzing text with the model..."):
            try:
                # Prepare the payload matching your FastAPI Pydantic schema
                payload = {"content": combined_text} 
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    # Assuming the API returns something like {"prediction": "positive", "probability": 0.95}
                    prediction = result.get("label", "Unknown")
                    confidence = result.get("confidence", 0.0)
                    st.divider()
                    st.subheader("Analysis Result")
                    # Display logic based on sentiment
                    if "positive" in str(prediction).lower():
                        st.success(f"**Sentiment:** {prediction.upper()}")
                        st.metric(label="Confidence Score", value=f"{confidence:.2%}")
                    else:
                        st.error(f"**Sentiment:** {prediction.upper()}")
                        st.metric(label="Confidence Score", value=f"{confidence:.2%}")
                else:
                    st.error(f"Error {response.status_code}: Could not get prediction.")
                    st.expander("Show Error Details").write(response.text)
            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Is the API running?")
                st.info("Make sure you started the backend with `uvicorn src.api.main:app --reload`")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# Sidebar info
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=150)
    st.markdown("### Model Information")
    st.info(
        """
        This interface connects to a FastAPI backend deployed on AWS.
        
        **Pipeline:**
        1. Text Preprocessing
        2. Tokenization
        2. Embeddings
        3. Model Inference
        """
    )
    st.caption("© 2025 MLOps Project Group")