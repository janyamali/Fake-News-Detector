import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit configuration
st.set_page_config(page_title="🧠 Fake News Verifier", layout="wide")
st.title("🧠 Fake News Summarizer & Verifier")

# Load summarizer model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Load fake news classification model
@st.cache_resource
def load_classifier():
    tokenizer = AutoTokenizer.from_pretrained("IIC/fake-news-detection-roberta")
    model = AutoModelForSequenceClassification.from_pretrained("IIC/fake-news-detection-roberta")
    return tokenizer, model

tokenizer, clf_model = load_classifier()

# User Input
st.subheader("Step 1: Paste a News Article")
article = st.text_area("📰 Enter the article text here:", height=300)

# Summarization
if st.button("🔍 Summarize Article"):
    if not article.strip():
        st.warning("⚠️ Please enter an article to summarize.")
    else:
        with st.spinner("Summarizing..."):
            try:
                summary = summarizer(article, max_length=250, min_length=50, do_sample=False)[0]['summary_text']
                st.subheader("📋 Summary:")
                st.success(summary)
            except Exception as e:
                st.error(f"Summarization failed: {e}")

# Fake News Detection
if st.button("🎯 Check Accuracy"):
    if not article.strip():
        st.warning("⚠️ Please paste an article first.")
    else:
        with st.spinner("Analyzing article..."):
            inputs = tokenizer(article, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                logits = clf_model(**inputs).logits
                probs = torch.softmax(logits, dim=1).squeeze()
            
            labels = ["FAKE", "REAL"]
            prediction = labels[torch.argmax(probs)]
            confidence = round(float(torch.max(probs)) * 100)

            st.subheader("📊 Prediction Result:")
            st.write(f"**Prediction:** `{prediction}`")
            st.write(f"**Confidence Score:** {confidence}/100")

            if prediction == "FAKE" and confidence < 70:
                if st.button("🛠 Suggest Corrected Version"):
                    with st.spinner("Using LLM to rewrite..."):
                        try:
                            prompt = f"""
                            You are a fact-checking assistant. The following article may contain misinformation.
                            Rewrite it using only factual, verifiable statements.

                            Article:
                            \"\"\"{article}\"\"\"

                            Corrected Version:
                            """
                            response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "You are a helpful and honest assistant."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.5,
                                max_tokens=800
                            )
                            corrected_text = response["choices"][0]["message"]["content"]
                            st.subheader("✅ Corrected Article:")
                            st.success(corrected_text)

                        except Exception as e:
                            st.error(f"LLM correction failed: {e}")
            elif prediction == "FAKE":
                st.error("🚨 The article appears to be fake.")
            else:
                st.success("✅ The article appears real and trustworthy.")
