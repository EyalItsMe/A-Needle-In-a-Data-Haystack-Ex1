import csv
import io
import os
import re
from os import listdir
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from festival import tfidf
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
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
        if word.isalnum() and word not in stop_words:
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
            magnitude1 = np.linalg.norm(list(doc1_tfidf.values()))
            magnitude2 = np.linalg.norm(list(doc2_tfidf.values()))
            if magnitude1 == 0 or magnitude2 == 0:
                row.append(0.0)  # Avoid division by zero
            else:
                cosine_similarity = dot_product / (magnitude1 * magnitude2)
                row.append(cosine_similarity)
        similarity_matrix.append(row)

    return similarity_matrix


def pagerank(similarity_matrix, damping_factor=0.01, max_iterations=10000, tolerance=1e-9):
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
    np_similarity_matrix = np.array(similarity_matrix)
    n = np_similarity_matrix.shape[0]

    ranks = np.ones(n) / n
    for _ in range(max_iterations):
        new_ranks = damping_factor * np.dot(np_similarity_matrix, ranks)
        new_ranks = new_ranks + (1 - damping_factor) / n

        if np.linalg.norm(new_ranks - ranks) < tolerance:
            return new_ranks

        ranks = new_ranks

    # Didn't converge, return the final ranks
    return ranks

def euclidean_distance(X, centroids):
    """
       Calculate the Euclidean distance between each data point in X and the centroids.

       Args:
           X (np.ndarray): The data points.
           centroids (np.ndarray): The centroids.

       Returns:
           np.ndarray: The Euclidean distances.
       """
    return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

def cosine_distance(X, centroids):
    """
        Calculate the cosine distance between each data point in X and the centroids.

        Args:
            X (np.ndarray): The data points.
            centroids (np.ndarray): The centroids.

        Returns:
            np.ndarray: The cosine distances.
        """
    similarities = cosine_similarity(X, centroids)
    return 1 - similarities  # Convert similarity to distance

# Modification: Combined distance metric that uses both cosine distance for TF-IDF vectors and Euclidean distance for numerical features.
def combined_distance(X_tfidf, X_features, centroids_tfidf, centroids_features, alpha=0.5):
    """
        Calculate a combined distance metric using both cosine distance for TF-IDF vectors
        and Euclidean distance for numerical features.

        Args:
            X_tfidf (np.ndarray): The TF-IDF feature vectors.
            X_features (np.ndarray): The numerical feature vectors.
            centroids_tfidf (np.ndarray): The centroids for TF-IDF features.
            centroids_features (np.ndarray): The centroids for numerical features.
            alpha (float, optional): The weight for combining distances. Defaults to 0.5.

        Returns:
            np.ndarray: The combined distances.
        """
    # Cosine distance for TF-IDF vectors
    cosine_dist = cosine_distance(X_tfidf, centroids_tfidf)
    # Euclidean distance for physical features
    euclidean_dist = euclidean_distance(X_features, centroids_features)
    # Combined distance
    combined_dist = alpha * cosine_dist + (1 - alpha) * euclidean_dist
    return combined_dist

def kmeans(X, k, distance_callback, iterations=100):
    """
       Perform k-means clustering.

       Args:
           X (np.ndarray): The data points.
           k (int): The number of clusters.
           distance_callback (callable): The distance function to use.
           iterations (int, optional): The number of iterations. Defaults to 100.

       Returns:
           tuple: Cluster labels and centroids.
       """
    centroids = X[np.random.choice(range(X.shape[0]), size=k, replace=False), :]
    for i in range(iterations):
        distances = distance_callback(X, centroids)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

# Modification: Combined k-means that utilizes both numerical and TF-IDF features
def kmeans_combined(X_num, X_tfidf, k, alpha=0.5, iterations=100):
    """
      Perform k-means clustering using both numerical and TF-IDF features.

      Args:
          X_num (np.ndarray): The numerical feature vectors.
          X_tfidf (np.ndarray): The TF-IDF feature vectors.
          k (int): The number of clusters.
          alpha (float, optional): The weight for combining distances. Defaults to 0.5.
          iterations (int, optional): The number of iterations. Defaults to 100.

      Returns:
          tuple: Cluster labels, numerical centroids, and TF-IDF centroids.
      """
    centroids_num = X_num[np.random.choice(range(X_num.shape[0]), size=k, replace=False), :]
    centroids_tfidf = X_tfidf[np.random.choice(range(X_tfidf.shape[0]), size=k, replace=False), :]

    for i in range(iterations):
        distances = combined_distance(X_num, X_tfidf, centroids_num, centroids_tfidf, alpha )
        labels = np.argmin(distances, axis=1)

        new_centroids_num = np.array([X_num[labels == j].mean(axis=0) for j in range(k)])
        new_centroids_tfidf = np.array([X_tfidf[labels == j].mean(axis=0) for j in range(k)])

        if np.all(centroids_num == new_centroids_num) and np.all(centroids_tfidf == new_centroids_tfidf):
            break
        centroids_num, centroids_tfidf = new_centroids_num, new_centroids_tfidf

    return labels, centroids_num, centroids_tfidf

def plot_kmeans(X, labels, title, k):
    """
      Plot the k-means clustering results.

      Args:
          X (np.ndarray): The data points.
          labels (np.ndarray): The cluster labels.
          title (str): The plot title.
          k (int): The number of clusters.
      """
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', label='Data points')
    plt.title(f'K-means Clustering ( k={k}) {title}')
    plt.xlabel('Amount of Sugar')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def preprocess_data(df):
    """
       Preprocess the data by mapping categorical features to numerical values.

       Args:
           df (pd.DataFrame): The input DataFrame.

       Returns:
           pd.DataFrame: The preprocessed DataFrame.
       """
    color_mapping = {'Black': 1, 'Blue': 2, 'Brown': 3, 'Green': 4, 'Orange': 5, 'Pink': 6, 'Purple': 7, 'Red': 8,
                     'Yellow': 9}
    peeling_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
    season_mapping = {'Fall': 1, 'Summer': 2, 'Winter': 3}

    df['Color'] = df['Color'].map(color_mapping)
    df['Peeling/Messiness'] = df['Peeling/Messiness'].map(peeling_mapping)
    df['Growth Season'] = df['Growth Season'].map(season_mapping)
    return df

def section_a(fruits):
    os.makedirs("fruit_json", exist_ok=True)
    for fruit in fruits:
        fruitcrawl(fruit)

def section_b(fruits):
    # section (b)
    fruit_data_folder = "fruit_json/"
    with io.open("fruits_summary.csv", "w", newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Fruit", "Summary"])

        for fruit in fruits:
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
            summary_text = " ".join(summary)

            csvwriter.writerow([fruit, summary_text])
            print(f"\nSummary for {fruit}:")
            for sentence in summary:
                print(f"- {sentence.strip('')}")

def section_c():
    summary_path = "fruits_summary.csv"
    df = pd.read_csv(summary_path)
    top_words = pd.DataFrame()

    for fruit, summary in df[['Fruit', 'Summary']].values:
        words = summary.replace(',', '').replace('-', ' ').lower().split(' ')
        unique_words = set(words)
        score = tfidf(summary_path, unique_words, False)
        # sort scores by the relevant fruit:
        fruit_scores = score.loc[fruit]
        top_fruit_words = fruit_scores.sort_values(ascending=False).head(3)
        #Get the row of the top fruit words:
        top_fruit = score[top_fruit_words.index]

        for word, value in top_fruit.items():
            if word not in top_words.columns:
                top_words.insert(len(top_words.columns), word, value)

    return top_words

def section_d(title):
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

    X = np.array(list(zip(sugar, time, price)))

    labels, centroids = kmeans(X, 3, euclidean_distance, 10000)

    X_plot = np.array(list(zip(sugar, price)))
    plot_kmeans(X_plot, labels,title,k=3)

def section_e(title):
    df = pd.read_csv("fruits.csv")
    X = preprocess_data(df)
    categorical_features = X[['Color', 'Peeling/Messiness', 'Growth Season']].values
    labels, centroids = kmeans(categorical_features, 3, euclidean_distance)
    X_plot = df[['Amount of Sugar', 'Price']].values
    plot_kmeans(X_plot, labels,title,k=3)

def section_f(top_words, title):
    df = pd.read_csv("fruits.csv")
    X = top_words.values
    labels, centroids = kmeans(X, 3,cosine_distance)

    X_plot = df[['Amount of Sugar', 'Price']].values
    plot_kmeans(X_plot, labels, title,k=3)

def section_g(top_words, title):
    df = pd.read_csv("fruits.csv")
    df = preprocess_data(df)
    X_num = df[['Amount of Sugar', 'Time it Lasts', 'Price']].values
    X_tfidf = top_words.values
    labels, centroids_num, centroids_tfidf = kmeans_combined(X_num, X_tfidf, k=3, alpha=0.5)
    X_plot = df[['Amount of Sugar', 'Price']].values
    plot_kmeans(X_plot, labels, title,k=3)

if __name__ == "__main__":
    fruits = extract_fruits("fruits.csv")
    section_a(fruits)
    section_b(fruits)
    top_words = section_c()
    section_d("based on Amount of Sugar, Time it lasts, Price")
    section_e("based on Color, Peeling Messiness, Growth Season")
    section_f(top_words, "based on tf-idf values of top 3 words")
    section_g(top_words, "based on all features")