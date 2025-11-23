# streamlit_app.py
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Anime Q&A using Gemini", layout="centered")

st.title("🎌 Anime Recommendation NLP System (Gemini)")
st.write("Ask any natural question about your anime dataset — like *Which is the most popular anime?*")

query = st.text_input("Enter your question:", "Which is the popular anime?")
top_k = st.slider("Number of retrieved results", 1, 10, 4)

if st.button("Get Answer"):
    payload = {"query": query, "top_k": top_k}
    try:
        with st.spinner("Getting answer from Gemini..."):
            res = requests.post(f"{API_URL}/answer", json=payload)
            if res.status_code == 200:
                data = res.json()
                st.subheader("💡 Answer")
                st.write(data["answer"])

                st.subheader("📚 Retrieved Sources")
                for s in data["sources"]:
                    st.markdown(f"**{s['name']}**")
                    st.write(s["text"][:400] + "...")
            else:
                st.error(f"Error: {res.status_code} - {res.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
