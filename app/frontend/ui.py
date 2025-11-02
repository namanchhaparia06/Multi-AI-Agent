import streamlit as st
import requests

from app.config.settings import settings
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

st.set_page_config(page_title="Multi AI Agent", layout="centered")
st.title("Multi AI Agent using Groq and Tavily")

system_prompt = st.text_area("Define your AI Agent:", height=70)
selected_model = st.selectbox("Select your AI Model: ", settings.ALLOWED_MODEL_NAMES)
allow_web_search = st.checkbox("Allow Web Search")
user_query = st.text_area("Enter your query: ", height=150)

API_URL = "http://127.0.0.1:9999/chat"

if st.button("Ask Agent") and user_query.strip():

    payload = {
        "model_name": selected_model,
        "system_prompt": system_prompt,
        "messages": [{"role": "user", "content": user_query}],  # <-- changed
        "allow_search": allow_web_search,
    }

    try:
        logger.info("Sending request to Backend")

        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            agent_response = response.json().get("response","")
            logger.info("Successfully received response from backend")

            st.subheader("Agent Response")
            st.markdown(agent_response.replace("\n","<br>"), unsafe_allow_html=True)

        else:
            logger.error("Failed to get response from backend")
            st.error(f"Error: {response.status_code} - {response.text}")
    
    except Exception as e:
        logger.error("Some error occured while connecting to backend")
        st.error("An error occurred while connecting to the backend.")
        raise CustomException("Failed to connect to backend", error_detail=e)