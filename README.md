# **Sentiment Analysis on Twitter Samples**

This code performs sentiment analysis using logistic regression on the Twitter samples dataset from NLTK. It trains a classifier to predict sentiment (positive or negative) based on the content of tweets.

## **Requirements**

- Python 3.x
- NLTK library

## **Setup**

1. Clone the repository:
    
    ```bash
    git clone https://github.com/chiemezie1/twitterSentimentAnalysis.git
    
    ```
    
2. Install NLTK if you haven't already:
    
    ```bash
    pip install nltk
    
    ```
    
3. Download NLTK's **`twitter_samples`** dataset by running the following Python code:
    
    ```python
    import nltk
    nltk.download('twitter_samples')
    ```
    
4. Run the main Python script:
    
    ```bash
    python sentiment_analysis_twitter.py
    ```
    

## **Usage**

The **`sentiment_analysis_twitter.py`** script performs the following tasks:

- Loads the Twitter samples dataset.
- Prepares the dataset for training and testing.
- Trains a logistic regression classifier using text vectorization.
- Tests the accuracy of the classifier.
- Predicts sentiment for sample tweets using the trained model.

## **Sample Output**

The script will output the accuracy of the classifier and predict sentiments for a set of sample tweets.
