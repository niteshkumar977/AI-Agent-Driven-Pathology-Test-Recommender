# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def preprocess_texts(texts):
    """
    Preprocess texts by joining all non-empty texts into a single string.

    :param texts: List of texts
    :return: List of preprocessed texts
    """
    processed_texts = []
    for text in texts:
        if isinstance(text, str) and text.strip():
            processed_texts.append(text)
    return " ".join(processed_texts)

def find_best_match(row, input_text):
    """
    Finds the best matched label for the given row's texts and labels.

    :param row: A single DataFrame row containing 'text' and 'label' columns
    :param input_text: The input text to match
    :return: Best matched label
    """
    # Join all non-empty texts into a single string
    texts = preprocess_texts(row.dropna().tolist())
    
    text_length = len(input_text)
    if text_length < 100:
        ngram_range = (2, 4)  # Short texts
    elif text_length < 250:
        ngram_range = (3, 6)  # Medium texts
    else:
        ngram_range = (4, 8)  # Long texts
    
    # Vectorize using character n-grams (to capture finer details)
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform([texts])
    tfidf_input_text = vectorizer.transform([input_text])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_input_text, tfidf_matrix)
    
    return cosine_sim[0][0]

def find_best_matches_from_dataframe(df, input_df):
    """
    Wrapper function to use the input DataFrame and find the best matches.

    :param df: DataFrame containing label and text columns
    :param input_df: DataFrame containing the input text in a column 'test_name'
    :return: DataFrame with best matched labels
    """
    input_texts = input_df['test_name'].tolist()
    best_matches = []

    for input_text in input_texts:
        # Calculate similarity for each row and find the best match
        df['similarity'] = df.apply(find_best_match, axis=1, input_text=input_text)
        best_match_label = df.loc[df['similarity'].idxmax(), 'label']
        best_matches.append(best_match_label)

    results_df = input_df.copy()
    results_df['best_match_label'] = best_matches

    return results_df

# Example usage:
if __name__ == "__main__":
    # Load the data from CSV files
    df = pd.read_csv('horizontal_input_df.csv', encoding='latin1')
    df.fillna('', inplace=True)

    # Input data for testing
    input_data = {
        "test_name": [
            "Thyroid Profile-Total (T3, T4 & TSH Ultra-sensitive)", "Lipid Profile", 
            "Kidney Function Test", "Progesterone (P4)", "COMPLETE BLOOD COUNT", 
            "IgE Total antibody", "Potassium, Serum", "Prolactin", "Random Blood Sugar", 
            "Testosterone Total"
        ]
    }

    input_df = pd.DataFrame(input_data)
    input_df['test_name'].fillna('', inplace=True)

    # Find best matches
    results_df = find_best_matches_from_dataframe(df, input_df)
    print(results_df)
# -


