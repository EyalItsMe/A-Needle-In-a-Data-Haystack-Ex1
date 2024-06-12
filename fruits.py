import csv
import io
import os
import re
from os import listdir
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd



import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

import requests
from bs4 import BeautifulSoup
import json
from nltk.corpus import stopwords  # Download stopwords corpus first
from nltk.tokenize import word_tokenize
from collections import defaultdict
from math import log


def fruitcrawl(fruit_name):
    """
    Crawls the Wikipedia page for a given fruit and saves the text content as a JSON object.

    Args:
        fruit_name: The name of the fruit (string).
    """
    wiki_fruits = ["Lime", "Date", "Orange", "Kiwi"]
    # Build the Wikipedia URL
    if fruit_name in wiki_fruits:
        wiki_url = f"https://en.wikipedia.org/wiki/{fruit_name}_(fruit)"
    else:
        wiki_url = f"https://en.wikipedia.org/wiki/{fruit_name}"

    # Try fetching the webpage content
    try:
        response = requests.get(wiki_url)
        response.raise_for_status()  # Raise exception for bad status codes
    except requests.exceptions.RequestException as e:
        print(f"Error fetching webpage for {fruit_name}: {e}")
        return

    # Parse the HTML content
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract all text content from paragraphs
    paragraphs = soup.find_all("p")
    text_content = "\n".join([p.get_text() for p in paragraphs])

    # Create the JSON object
    fruit_data = {
        "fruit_name": fruit_name,
        "text_content": text_content
    }
    filename = f"fruit_json/{fruit_name}.json"

    with open(filename, "w") as f:
        json.dump(fruit_data, f, indent=4)
        print(f"Successfully saved data for {fruit_name} to {filename}")


def extract_fruits(csv_file):
    fruits = []
    with open(csv_file, "r") as f:
        # Skip the header row (assuming the first row contains column names)
        next(f)  # Read and discard the first line
        for line in f:
            fruit_name = line.split(",")[0].strip()
            fruits.append(fruit_name)
    return fruits


def load_fruit_docs(folder_path):
    """Loads text content from all JSON files in a folder."""
    documents = []
    for filename in listdir(folder_path):
        if filename.endswith(".json"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r") as f:
                fruit_data = json.load(f)
                text_content = fruit_data["text_content"]
                documents.append(text_content)
    return documents


def clean_and_split_text(text):
    # Remove references like [7], [31], etc.
    text = re.sub(r'\[\d+\]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Split text into sentences
    return text.split(". ")


def tf_idf(text, documents):
    """
    Calculates TF-IDF scores for words in a text.

    Args:
        text: The text string to calculate TF-IDF for.
        documents: A list of strings representing the documents (in this case, the content from JSON files).

    Returns:
        A dictionary mapping each word to its TF-IDF score.
    """

    stop_words = set(stopwords.words('english'))
    word_counts = defaultdict(int)  # Count word occurrences
    total_words = 0

    # Tokenize and filter stop words
    for word in word_tokenize(text.lower()):
        if word not in stop_words:
            word_counts[word] += 1
            total_words += 1

    # Calculate TF and IDF components
    tf_idfs = {}
    for word, count in word_counts.items():
        tf = count / total_words
        idf = log(1 + len(documents)) / (1 + sum([1 for doc in documents if word in doc]))
        tf_idfs[word] = tf * idf

    return tf_idfs


def build_similarity_matrix(documents):
    """
    Builds a similarity matrix between documents using TF-IDF features.

    Args:
        documents: A list of text strings representing documents.

    Returns:
        A 2D list representing the similarity matrix.
    """
    similarity_matrix = []
    tfidf_all = [tf_idf(doc, documents) for doc in documents]  # Calculate TF-IDF for each doc

    # Calculate cosine similarity between each document pair
    for i in range(len(documents)):
        row = []
        for j in range(len(documents)):
            if i == j:
                row.append(1.0)  # Same document has similarity 1
                continue
            doc1_tfidf = tfidf_all[i]
            doc2_tfidf = tfidf_all[j]
            dot_product = sum(tfidf1 * tfidf2 for tfidf1, tfidf2 in zip(doc1_tfidf.values(), doc2_tfidf.values()))
            magnitude1 = sum(tfidf ** 2 for tfidf in doc1_tfidf.values()) ** 0.5
            magnitude2 = sum(tfidf ** 2 for tfidf in doc2_tfidf.values()) ** 0.5
            if magnitude1 == 0 or magnitude2 == 0:
                row.append(0.0)  # Avoid division by zero
            else:
                cosine_similarity = dot_product / (magnitude1 * magnitude2)
                row.append(cosine_similarity)
        similarity_matrix.append(row)

    return similarity_matrix


def pagerank(similarity_matrix, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
    """
    Implements PageRank algorithm on the similarity matrix.

    Args:
        similarity_matrix: A 2D list representing the similarity matrix.
        damping_factor: The damping factor for PageRank (default: 0.85).
        max_iterations: Maximum number of iterations for PageRank (default: 100).
        tolerance: Convergence tolerance for PageRank (default: 1e-6).

    Returns:
        A list of PageRank scores for each document (sentence).
    """
    n = len(similarity_matrix)
    ranks = [1.0 / n for _ in range(n)]  # Initialize with uniform ranks

    for _ in range(max_iterations):
        new_ranks = [damping_factor * sum(similarity * rank for similarity, rank in zip(row, ranks)) for row in
                     similarity_matrix]
        new_ranks = [new_rank + (1 - damping_factor) / n for new_rank in new_ranks]  # Add random surfer weight

        # Check for convergence
        if sum(abs(new_rank - old_rank) for new_rank, old_rank in zip(new_ranks, ranks)) < tolerance:
            return new_ranks

        ranks = new_ranks

    # Didn't converge, return the final ranks
    return ranks


def summarize_fruit(fruit_name, fruit_data_folder):
    """
    Summarizes the text content for a given fruit using TF-IDF and PageRank.

    Args:
        fruit_name: The name of the fruit (string).
        fruit_data_folder: The folder path containing JSON files (string).
    """
    # Load text content from the JSON file
    with open(f"{fruit_data_folder}{fruit_name}.json", "r") as f:
        fruit_data = json.load(f)
        text_content = fruit_data["text_content"]

    # Split text into sentences
    sentences = text_content.split(". ")

    # Build a similarity matrix using TF-IDF and calculate PageRank scores
    similarity_matrix = build_similarity_matrix(sentences)
    sentence_ranks = pagerank(similarity_matrix)

    # Extract the top 5 sentences as a summary
    summary = [sentence for sentence, rank in sorted(zip(sentences, sentence_ranks), key=lambda x: x[1], reverse=True)[:5]]

    print(f"\nSummary for {fruit_name}:")
    for sentence in summary:
        print(f"- {sentence.strip()}")

def kmeans(X,k, iterations=1000):
    centroids = X[np.random.choice(range(X.shape[0]), size=k, replace=False), :]

    for i in range(iterations):
        distances = np.sqrt(((X-centroids[:,np.newaxis])**2).sum(axis=2))

        labels = np.argmin(distances, axis=0)

        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        if np.all(centroids == new_centroids):
            break
    return labels, centroids


def plot_kmeans(X, labels, centroids):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', label='Data points')
    plt.title('K-means Clustering of Fruits')
    plt.xlabel('Amount of Sugar')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def kmeans_mixed(X, k, iterations=100):
    centroids = X[np.random.choice(range(X.shape[0]), size=k, replace=False), :]

    for i in range(iterations):
        distances = np.array([[np.linalg.norm(x - centroid) for centroid in centroids] for x in X])
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

def preprocess_data(df):
    # Label encoding
    color_mapping = {'Black': 1, 'Blue': 2, 'Brown': 3, 'Green': 4, 'Orange': 5, 'Pink': 6, 'Purple': 7, 'Red': 8,
                     'Yellow': 9}
    peeling_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
    season_mapping = {'Fall': 1, 'Summer': 2, 'Winter': 3}

    df['Color'] = df['Color'].map(color_mapping)
    df['Peeling/Messiness'] = df['Peeling/Messiness'].map(peeling_mapping)
    df['Growth Season'] = df['Growth Season'].map(season_mapping)
    return df


def section_b():
    # section (b)
    fruit_data_folder = "fruit_json/"
    # Load all fruit text data from the JSON files
    fruit_documents = load_fruit_docs(fruit_data_folder)
    # Open the CSV file for writing with UTF-8 encoding
    with io.open("fruits_summary.csv", "w", newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header row
        csvwriter.writerow(["Fruit", "Summary"])

        for fruit in fruits:
            # Get the fruit's text content from the JSON file
            with open(f"{fruit_data_folder}{fruit}.json", "r", encoding='utf-8') as f:
                fruit_data = json.load(f)
                text_content = fruit_data["text_content"]

            # Split text into sentences
            sentences = clean_and_split_text(text_content)

            # Build the similarity matrix
            similarity_matrix = build_similarity_matrix(sentences)

            # Calculate PageRank scores for sentences
            sentence_ranks = pagerank(similarity_matrix)

            # Extract the top 5 sentences as a summary
            summary = [sentence for sentence, rank in
                       sorted(zip(sentences, sentence_ranks), key=lambda x: x[1], reverse=True)[:5]]

            # Join the summary sentences into a single string
            summary_text = " ".join(summary)

            # Write the fruit name and summary to the CSV file
            csvwriter.writerow([fruit, summary_text])

            print(f"\nSummary for {fruit}:")
            for sentence in summary:
                print(f"- {sentence.strip('')}")

def section_d():
    # section d:
    # Load the amount of sugar, time it last and price from fruits.csv
    fruits = []
    with open("fruits.csv", "r") as f:
        next(f)
        for line in f:
            fruit_name, _, _, _, price, sugar, time = line.split(",")
            fruits.append([fruit_name, float(sugar), float(time), float(price)])

    # Normalize the sugar, time and price
    sugar = np.array([fruit[1] for fruit in fruits])
    time = np.array([fruit[2] for fruit in fruits])
    price = np.array([fruit[3] for fruit in fruits])

    # Combine the normalized sugar, time and price
    X = np.array(list(zip(sugar, time, price)))

    # Perform K-means clustering
    labels, centroids = kmeans(X, 3, 10000)

    X_plot = np.array(list(zip(sugar, price)))
    # Plot the K-means clustering
    plot_kmeans(X_plot, labels, centroids)

def section_e():
    # Load the data from CSV file
    df = pd.read_csv("fruits.csv")
    X = preprocess_data(df)

    categorical_features = X[['Color', 'Peeling/Messiness', 'Growth Season']].values

    labels, centroids = kmeans_mixed(categorical_features, 3)

    X_plot = df[['Amount of Sugar', 'Price']].values
    plot_kmeans(X_plot, labels, centroids)

if __name__ == "__main__":
    # section (a)
    fruits = extract_fruits("fruits.csv")
    # for fruit in fruits:
    #     fruitcrawl(fruit)

    # section_b()
    # section_d()
    section_e()