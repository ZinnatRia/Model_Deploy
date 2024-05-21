import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import string
from cleantext import clean
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Preprocessing functions
def remove_repeated_punctuation(text):
    cleaned_text = []
    last_char = None
    for char in text:
        if char in string.punctuation:
            if char != last_char:
                cleaned_text.append(char)
            last_char = char
        else:
            cleaned_text.append(char)
            last_char = None
    return ''.join(cleaned_text)

def clean_text(text):
    return clean(text,
        fix_unicode=True,
        to_ascii=True,
        lower=True,
        no_line_breaks=False,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=True,
        no_currency_symbols=True,
        no_punct=True,
        replace_with_punct="<punc>",
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        replace_with_number="<NUMBER>",
        replace_with_currency_symbol="<CUR>",
        lang="en"
    )

# Function to remove less important words
def remove_less_important_words(text, threshold=0.00004):
    # Create a DataFrame from user input for the TF-IDF process
    train_df = pd.DataFrame({'text': [text]})
    
    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer()

    # Fit and transform the text data
    tfidf_matrix = tfidf.fit_transform(train_df['text'])

    # Get feature names (words)
    feature_names = tfidf.get_feature_names_out()

    # Create a DataFrame of TF-IDF scores
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # Find mean TF-IDF score for each word across all documents
    word_scores = tfidf_df.mean(axis=0)

    # Get less important words based on threshold
    less_important_words = word_scores[word_scores < threshold]

    # Remove less important words from text
    words = text.split()
    filtered_words = [word for word in words if word not in less_important_words]
    return ' '.join(filtered_words)

# Combine preprocessing steps
def preprocess_text(text):
    text = remove_repeated_punctuation(text)
    text = clean_text(text)
    text = remove_less_important_words(text)
    return text

# Cache the model and tokenizer
@st.cache_resource
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained("ZInnatRia/ClinicalText")
    return tokenizer, model

tokenizer, model = get_model()

# Add a title and description to the app
st.title("Clinical Text Mortality Prediction")
st.markdown("""
This app uses a fine-tuned BERT model to predict mortality based on clinical text input.
Please enter clinical text below and click the "Predict" button to see the prediction.
""")

# Create a sidebar with additional information
st.sidebar.header("About")
st.sidebar.markdown("""
This application is a demonstration of a BERT model fine-tuned for mortality prediction using clinical text.
""")

# Get user input
user_input = st.text_area('Enter Clinical Text for Mortality Prediction', height=400)
button = st.button("Predict")

# Define the prediction labels
d = {
    0: 'neoplasms',
    1: 'digestive system diseases',
    2: 'nervous system diseases',
    3: 'cardiovascular diseases',
    4: 'general pathological conditions' 
}

# Perform prediction when button is clicked
if user_input and button:
    with st.spinner('Predicting...'):
        try:
            # Show raw user input
            st.write("Raw User Input:")
            st.text_area("Raw Text", user_input, height=200, max_chars=None, key=None)

            # Preprocess text
            preprocessed_text = preprocess_text(user_input)

            # Display preprocessed text
            st.write("Preprocessed Text:")
            st.text_area("Preprocessed Text", preprocessed_text, height=200, max_chars=None, key=None)

            # Tokenize and predict
            test_sample = tokenizer([preprocessed_text], padding=True, truncation=True, max_length=512, return_tensors='pt')
            output = model(**test_sample)
            st.write("Logits: ", output.logits)
            y_pred = np.argmax(output.logits.detach().numpy(), axis=1)
            st.success(f"Prediction: {d[y_pred[0]]}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Add custom CSS for wider text area
st.markdown("""
<link rel="stylesheet" type="text/css" href="style.css">
""", unsafe_allow_html=True)
