import streamlit as st
import spacy
from langdetect import detect
from indicnlp.tokenize import sentence_tokenize
from transformers import pipeline
import networkx as nx
import matplotlib.pyplot as plt

# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")

# Load Pegasus summarizer
summarizer = pipeline("summarization", model="google/pegasus-xsum")

# Initialize zero-shot classifier for validation
classifier = pipeline("zero-shot-classification")

# Function for language detection
def detect_language(text):
    return detect(text)

# Function for tokenization (sentence and paragraph)
def tokenize_text(text):
    sentences = sentence_tokenize.sentence_split(text)
    paragraphs = text.split("।")
    return sentences, paragraphs

# Function for calculating dilemma score based on keywords
def calculate_dilemma_score(text):
    score = 0
    if "should" in text and "but" in text:
        score += 3
    if "happy" in text and "sad" in text:
        score += 3
    moral_keywords = ["duty", "truth", "betrayal", "dharma"]
    for keyword in moral_keywords:
        if keyword in text:
            score += 2
    if "I don't know what to do" in text:
        score += 4
    return score

# Function for dependency parsing and displaying results
def dependency_parsing(text):
    doc = nlp(text)
    dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
    return dependencies

# Function for summarizing dilemma passages
def summarize_dilemma(text):
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    return summary[0]['summary_text']

# Function for creating a knowledge graph
def create_knowledge_graph():
    G = nx.Graph()
    G.add_node("Rama", type="Character")
    G.add_node("Loyalty", type="Moral Value")
    G.add_node("Duty", type="Moral Value")
    G.add_node("Conflict", type="Conflict")
    G.add_edge("Rama", "Loyalty", relationship="values")
    G.add_edge("Rama", "Duty", relationship="torn between")
    G.add_edge("Loyalty", "Duty", relationship="in conflict")
    
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color="skyblue", node_size=3000, font_size=12)
    plt.title("Moral Dilemma Knowledge Graph")
    st.pyplot(plt)

# Main function for Streamlit UI
def main():
    st.title("Moral Dilemma Analysis Tool")
    st.markdown("Enter the text below and see the results of moral dilemma analysis.")

    # Input text box
    input_text = st.text_area("Enter your text here:")

    if input_text:
        # Step 1: Language Detection
        language = detect_language(input_text)
        st.subheader(f"Detected Language: {language}")

        # Step 2: Tokenization
        sentences, paragraphs = tokenize_text(input_text)
        st.subheader("Tokenized Sentences:")
        st.write(sentences)
        st.subheader("Tokenized Paragraphs:")
        st.write(paragraphs)

        # Step 3: Dilemma Score Calculation
        dilemma_score = calculate_dilemma_score(input_text)
        st.subheader(f"Dilemma Score: {dilemma_score}")

        # Step 4: Dependency Parsing
        dependencies = dependency_parsing(input_text)
        st.subheader("Dependency Parsing:")
        st.write(dependencies)

        # Step 5: Summarization
        summary = summarize_dilemma(input_text)
        st.subheader("Summary of Dilemma Passage:")
        st.write(summary)

        # Step 6: Knowledge Graph Visualization
        st.subheader("Knowledge Graph:")
        create_knowledge_graph()

        # Step 7: Zero-Shot Classification
        result = classifier(input_text, candidate_labels=["moral dilemma", "non-dilemma"])
        st.subheader(f"Zero-Shot Classification Result: {result}")

if __name__ == "__main__":
    main()
