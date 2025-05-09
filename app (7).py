
import streamlit as st
from langdetect import detect
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')

# Title
st.title("🧠 Moral Dilemma Analyzer")

# Text input
text = st.text_area("Paste your chapter or paragraph here", height=300)

if st.button("Analyze"):
    if text:
        # Step 1: Language Detection
        lang = detect(text)
        st.write(f"🔍 Detected Language: `{lang}`")

        # Step 2: Tokenization
        sentences = sent_tokenize(text)
        st.write(f"🧾 Sentence Count: {len(sentences)}")

        # Sample Heuristic Scoring
        dilemma_keywords = ["should", "must", "either", "or", "duty", "justice", "truth", "torn", "conflict", "betrayal"]
        score = sum(1 for s in sentences for word in dilemma_keywords if word in s.lower())
        st.write(f"⚖️ Dilemma Score: {score}")

        if score >= 7:
            st.success("✅ Likely Moral Dilemma Detected!")
        else:
            st.info("ℹ️ No strong dilemma indicators found.")
    else:
        st.warning("Please enter text to analyze.")
