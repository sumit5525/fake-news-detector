# Fake News Detector 📰

## Overview
A Machine Learning project that classifies news articles 
as Real or Fake using Natural Language Processing (NLP). 
Built using TF-IDF Vectorization and Logistic Regression 
achieving high accuracy on unseen data.

## Colab link
https://colab.research.google.com/drive/12kCSsoR2gJtEyE7tMZdHCMVEkPAopuGP

## Dataset
- source: kaggle
- Format: CSV
- Columns: ID, Title, Author, Text, Label
- Label: 0 = Real News, 1 = Fake News

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- NLTK (PorterStemmer, Stopwords)

## Workflow
1. Load and clean dataset
2. Merge Title + Author into single content feature
3. Text preprocessing:
   - Lowercase conversion
   - Remove non-alphabetical characters
   - Stop word removal
   - Stemming via PorterStemmer
4. TF-IDF Vectorization
5. Stratified Train-Test Split (80-20)
6. Train Logistic Regression model
7. Evaluate accuracy on train and test sets

## Key Concepts
- TF-IDF: Converts text to numerical vectors
- Stemming: Reduces words to root form
- Stratification: Maintains class balance in split
- Logistic Regression: Binary classification using Sigmoid function

## Results
- High accuracy on both training and test sets
- Model generalizes well with no overfitting
