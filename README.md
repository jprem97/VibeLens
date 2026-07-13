# Flipkart Reviews Sentiment Analysis

A beginner-friendly Python project for sentiment analysis of Flipkart product reviews. This project classifies reviews into **Positive**, **Neutral**, and **Negative** sentiments using machine learning models. It also includes **sarcasm detection** to identify potentially sarcastic reviews.

## Project Overview

This project analyzes Flipkart product reviews and predicts their sentiment based on the review text. It uses:
- **TF-IDF** for feature extraction
- **Naive Bayes** and **Logistic Regression** for sentiment classification
- **Logistic Regression** for sarcasm detection
- **Word Clouds** and **Bar Charts** for visualization

## Features

- Modular code structure with separate files for each task
- Interactive sentiment predictor
- Sarcasm detection with confidence score
- Word cloud generation for positive and negative reviews
- Confusion matrix visualization
- Model comparison table
- Analytical results with top words analysis

## Folder Structure

```
Flipkart-Sentiment-Analysis/
│
├── data/
│   ├── flipkart_data.csv
│   └── sarcastic_flipkart_reviews_10000.csv
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── model.py
│   ├── sarcasm_model.py
│   ├── predictor.py
│   └── main.py
│
├── sentiment/
│   └── sentiment_analysis.ipynb
│
├── output/
│   ├── positive_wordcloud.png
│   ├── negative_wordcloud.png
│   ├── sentiment_distribution.png
│   ├── confusion_matrix_nb.png
│   ├── confusion_matrix_lr.png
│   └── model_comparison.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Flipkart-Sentiment-Analysis.git
cd Flipkart-Sentiment-Analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Required Libraries

- pandas
- numpy
- matplotlib
- seaborn
- wordcloud
- nltk
- scikit-learn
- jupyter

## How to Run

Run the main script:
```bash
python src/main.py
```

This will:
- Load the dataset
- Perform EDA
- Clean the reviews
- Generate visualizations
- Train both models
- Evaluate and compare models
- Start interactive prediction (with sarcasm detection)

## How to Run Notebook

1. Open Jupyter Notebook:
```bash
jupyter notebook
```

2. Navigate to `sentiment/sentiment_analysis.ipynb`

3. Run all cells to see the complete pipeline

## Expected Outputs

After running the project, you will get:

- **Sentiment Distribution Chart** - Bar chart showing review counts
- **Word Clouds** - Visual representation of frequent words
- **Confusion Matrices** - For both Naive Bayes and Logistic Regression
- **Model Comparison Table** - Accuracy, Precision, Recall, F1 Score
- **Interactive Predictor** - Test your own reviews with sentiment and sarcasm detection

## Sample Prediction

```
=================================
Flipkart Review Sentiment Checker
=================================

Enter your review:
This phone is amazing.

Predicted Sentiment : Positive
Sarcasm Detected    : Not Sarcastic (Confidence: 85.2%)

Would you like to test another review? (Y/N):
```

## Screenshots

### Sentiment Distribution
![Sentiment Distribution](output/sentiment_distribution.png)

### Positive Word Cloud
![Positive Word Cloud](output/positive_wordcloud.png)

### Negative Word Cloud
![Negative Word Cloud](output/negative_wordcloud.png)

### Confusion Matrix - Naive Bayes
![Confusion Matrix NB](output/confusion_matrix_nb.png)

### Confusion Matrix - Logistic Regression
![Confusion Matrix LR](output/confusion_matrix_lr.png)
