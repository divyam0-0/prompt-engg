import streamlit as st 
from langchain.callbacks.streamlit import StreamlitCallbackHandler

st.set_page_config(
    page_title="Prompt Engineering",
)

from functions import (
    improve_prompt,
    answer_prompt,
    combine_answers
)

#import mlflow
sidebar_message = """
ML Ops Capstone Project


By:
Divyam Bansal

Himanshu Srivastava

Vignesh N
"""

import streamlit as st
import plotly.express as px
# import plotly.graph_objects as go
from textstat import flesch_reading_ease
import syllapy
import tiktoken
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Ensure NLTK resources are available
nltk.download("punkt_tab")
nltk.download("stopwords")

tokenizer = tiktoken.get_encoding("cl100k_base")  # Uses OpenAI's tokenizer

def get_token_length(text):
    """Compute token length using tiktoken."""
    return len(tokenizer.encode(text))

def calculate_metrics(text):
    """Compute multiple text metrics including token length."""
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    num_chars = sum(len(word) for word in words)
    num_words = len(words)
    num_sentences = len(sentences)
    num_unique_words = len(set(words))
    num_syllables = sum(syllapy.count(word) for word in words)

    if num_words == 0 or num_sentences == 0:
        return None

    ari = (4.71 * (num_chars / num_words)) + (0.5 * (num_words / num_sentences)) - 21.43
    lexical_diversity = num_unique_words / num_words
    flesch_score = flesch_reading_ease(text)
    ttr = num_unique_words / num_words  # Type-Token Ratio (TTR)
    avg_word_length = num_chars / num_words  # Average Word Length
    token_length = get_token_length(text)  # Token count from tiktoken

    return {
        "ARI": round(ari, 2),
        "Lexical Diversity": round(lexical_diversity, 2),
        "Flesch Reading Ease": round(flesch_score, 2),
        "Average Word Length": round(avg_word_length, 2),
        "Type-Token Ratio (TTR)": round(ttr, 2),
        "Token Length": token_length
    }


with st.sidebar:
    st.write(sidebar_message)
    

expander = st.expander("Tips")
expander.write("Try running any question on your mind. The app will try to answer it, and then improve the prompt to see if it can answer it better. For example, _help me understand prompt engineering_.")

# tab_1, tab_2, tab_3, tab_4 = st.tabs(['Cost','Model',"📈 Prompt", "🗃 Transaction"])

if user_input := st.chat_input("Ask anything"):

    tab_1, tab_2, tab_3, tab_4 = st.tabs([
            '👤 User', 
            '🚀 Improved Prompt', 
            '🔍 Comparison', 
            '📊 Graph'
        ])

    with tab_1:
        st.chat_message("user").write(user_input)
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            # a1 = answer_prompt(user_input, callbacks=[st_callback], system_instructions="")
            a1 = answer_prompt(user_input, system_instructions="", callbacks=[st_callback])
            # st.write(a1)
    with tab_2:
        st.info("**Improved prompt** \n\n The app will now try to improve your prompt.")
        with st.chat_message("user"):
            st_callback = StreamlitCallbackHandler(st.container())
            new_prompt_complex = improve_prompt(user_input, simple_instruction=False, use4 = False, callbacks=[st_callback])
            # st.write(new_prompt_complex)
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            a_complex = answer_prompt(new_prompt_complex, callbacks=[st_callback])
            # st.write(a_complex)
    with tab_3:
        with st.chat_message("user"):
            st.write("Describe the difference between these two answers and summarize them into a single answer.")

        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            combined = combine_answers([a1,a_complex],
                                        user_input,
                                        callbacks=[st_callback]
                                        )
            # st.write(combined)
    with tab_4:
        elements = {
        "User Prompt": user_input,
        "AI Output": a1,
        "Engineered AI Prompt": new_prompt_complex,
        "New AI Output": a_complex}
        data = {name: calculate_metrics(text) for name, text in elements.items()}
        df = pd.DataFrame([
        {"Element": name, "Metric": metric, "Value": value}
        for name, metrics in data.items()
        for metric, value in metrics.items()
            ])

        st.title("Prompt Engineering - Text Metrics Analysis")

        # ARI - Bar Chart
        st.subheader("Automated Readability Index (ARI)")
        fig_ari = px.bar(df[df["Metric"] == "ARI"], x="Element", y="Value", color="Element",
                        title="Automated Readability Index (ARI) Comparison")
        st.plotly_chart(fig_ari)

        # Lexical Diversity - Box Plot
        st.subheader("Lexical Diversity")
        fig_lex = px.box(df[df["Metric"] == "Lexical Diversity"], x="Element", y="Value", color="Element",
                        title="Lexical Diversity Distribution")
        st.plotly_chart(fig_lex)

        # Flesch Reading Ease - Bar Chart
        st.subheader("Flesch Reading Ease")
        fig_fre = px.bar(df[df["Metric"] == "Flesch Reading Ease"], x="Element", y="Value", color="Element",
                        title="Flesch Reading Ease Score Comparison")
        st.plotly_chart(fig_fre)

        # Average Word Length - Histogram
        st.subheader("Average Word Length")
        fig_word_length = px.histogram(df[df["Metric"] == "Average Word Length"], x="Element", y="Value",
                                    color="Element", nbins=4, title="Average Word Length Distribution")
        st.plotly_chart(fig_word_length)

        # Type-Token Ratio (TTR) - Line Chart
        st.subheader("Type-Token Ratio (TTR)")
        fig_ttr = px.line(df[df["Metric"] == "Type-Token Ratio (TTR)"], x="Element", y="Value", markers=True,
                        title="Type-Token Ratio (TTR) Trend")
        st.plotly_chart(fig_ttr)

        st.subheader("Token Length")
        fig_token = px.bar(df[df["Metric"] == "Token Length"], x="Element", y="Value", color="Element",
                        title="Token Length using tiktoken")
        st.plotly_chart(fig_token)
        # runs = mlflow.search_traces()
        # runs_df = pd.DataFrame(runs.values, columns=runs.columns, index=runs.index)
        # runs_df.to_csv('mlflow_traces.csv', index=False,)
