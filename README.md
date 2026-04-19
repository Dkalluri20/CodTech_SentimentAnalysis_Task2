# CodTech_SentimentAnalysis_Task2
# Task 2: Sentiment Analysis on Customer Reviews 🎬

## 📌 Project Overview
The objective of this task was to build a robust Natural Language Processing (NLP) pipeline to classify the sentiment of movie reviews. This project demonstrates the transition from basic text classification to an advanced **Ensemble Learning** approach to achieve high predictive accuracy.

## 🛠️ Technical Workflow

### 1. Data Preprocessing
* **Cleaning:** Used Regular Expressions (RegEx) to strip HTML tags, punctuation, and numerical noise.
* **Normalization:** Implemented **WordNet Lemmatization** to reduce words to their dictionary root (e.g., "better" and "good" are linked), reducing feature sparsity.

### 2. Feature Engineering (TF-IDF)
* **Vectorization:** Applied `TfidfVectorizer` to convert text into numerical vectors.
* **N-gram Analysis:** Utilized a range of **(1, 3)** n-grams. This allows the model to capture:
    * **Unigrams:** "good", "bad"
    * **Bigrams:** "not good", "very happy"
    * **Trigrams:** "not worth watching", "absolutely loved it"
* **Sublinear Scaling:** Applied logarithmic frequency scaling to prevent high-frequency "filler" words from dominating the model.

### 3. Model Architecture (Ensemble Voting)
To maximize accuracy, a **Voting Classifier** was implemented, combining three distinct algorithms:
1. **Linear Support Vector Machine (SVC):** For finding the optimal hyperplane in high-dimensional text space.
2. **Logistic Regression:** For probabilistic classification.
3. **Multinomial Naive Bayes:** A classic baseline for frequency-based text analysis.

## 📊 Results & Visualization
* **Accuracy:** Successfully achieved high-performance metrics (targeting 90%+).
* **Confusion Matrix:** Included to visualize the precision and recall across Positive and Negative classes.
* **Word Cloud:** Generated to highlight the most frequent descriptors used in positive sentiment reviews.

## 🚀 How to Run
1. Open `CODTECH-Task2-Advanced-Sentiment-Analysis.ipynb` in Google Colab.
2. Ensure `nltk`, `scikit-learn`, and `wordcloud` are installed.
3. Run all cells to download the NLTK `movie_reviews` dataset and train the ensemble model.
