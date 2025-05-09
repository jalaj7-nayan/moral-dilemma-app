import streamlit as st
from langdetect import detect
import nltk
import spacy
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from transformers import pipeline

# Download models if not already present
nltk.download('punkt')
try:
    nlp = spacy.load("en_core_web_sm")
except:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Title
st.title("🧠 Moral Dilemma Detector")

# Text Input
user_text = st.text_area("Enter a chapter or paragraph:", height=300)

if st.button("Analyze"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        # Step 1: Language Detection
        lang = detect(user_text)
        st.markdown(f"**Detected Language:** `{lang}`")

        # Step 2: Tokenization
        sentences = nltk.sent_tokenize(user_text)
        st.markdown(f"**Number of Sentences:** {len(sentences)}")

        # Step 3: Cognitive Tension Scoring
        def calculate_score(text):
            score = 0
            if any(x in text.lower() for x in ["should", "must", "but", "or"]):
                score += 3
            if any(x in text.lower() for x in ["justice", "duty", "truth", "betrayal", "sin", "dharma"]):
                score += 2
            if "i don't know" in text.lower() or "i was torn" in text.lower():
                score += 4
            if any(x in text.lower() for x in ["either", "if", "then", "else"]):
                score += 2
            if any(x in text.lower() for x in ["however", "yet", "although", "despite"]):
                score += 1
            return score

        scored = [{"sentence": s, "score": calculate_score(s)} for s in sentences]
        df = pd.DataFrame(scored)
        st.dataframe(df)

        dilemma_zones = df[df["score"] >= 7]
        if not dilemma_zones.empty:
            st.success("🎯 Possible Moral Dilemmas Found:")
            for i, row in dilemma_zones.iterrows():
                st.markdown(f"✅ **Sentence:** {row['sentence']}")
                st.markdown(f"🔢 **Score:** {row['score']}")
        else:
            st.info("No moral dilemmas detected with current thresholds.")

        # Optional: Visual Knowledge Graph
        G = nx.Graph()
        for i, row in dilemma_zones.iterrows():
            G.add_node(f"Sentence {i+1}", label=row['sentence'])
            G.add_node(f"Score {i+1}", label=f"Score: {row['score']}")
            G.add_edge(f"Sentence {i+1}", f"Score {i+1}")

        if G.number_of_nodes() > 0:
            st.markdown("### 📊 Knowledge Graph")
            plt.figure(figsize=(10, 5))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
            st.pyplot(plt)
