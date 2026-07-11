"""
predictor.py
Interactive sentiment predictor that uses the trained model.
"""

from src.preprocessing import clean_text


def predict_sentiment(tfidf, model, review):
    """
    Predict sentiment for a single review.
    Uses the same TF-IDF vectorizer (transform only, not fit_transform).
    """
    cleaned = clean_text(review)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)
    return prediction[0]


def interactive_predictor(tfidf, model):
    """
    Interactive loop: ask user for reviews, predict sentiment,
    and repeat until user enters N.
    """
    print("\n=================================")
    print("Flipkart Review Sentiment Checker")
    print("=================================")

    while True:
        review = input("\nEnter your review:\n")

        sentiment = predict_sentiment(tfidf, model, review)
        print(f"\nPredicted Sentiment : {sentiment}")

        choice = input("\nWould you like to test another review? (Y/N): ").strip().upper()
        if choice != "Y":
            print("Thank you for using Sentiment Checker!")
            break
