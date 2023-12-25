import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download the twitter_samples dataset
nltk.download('twitter_samples')

# Import twitter_samples dataset
from nltk.corpus import twitter_samples

# Load positive and negative tweets
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# Creating labelled data
documents = []

# Adding positive tweets
for tweet in positive_tweets:
    documents.append((tweet, "positive"))

# Adding negative tweets
for tweet in negative_tweets:
    documents.append((tweet, "negative"))

# Split the dataset into the text and labels
texts, labels = zip(*documents)

# Split data into training and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.25, random_state=42)

# Begin text vectorization
vectorizer = CountVectorizer(ngram_range=(1, 2))

# Fit and transform the training data
train_vectors = vectorizer.fit_transform(train_texts)

# Transform the test data
test_vectors = vectorizer.transform(test_texts)

# Initialize the Logistic Regression classifier
logistic_classifier = LogisticRegression()

# Train the classifier
logistic_classifier.fit(train_vectors, train_labels)

# Predict sentiments for test data using the trained classifier
logistic_predictions = logistic_classifier.predict(test_vectors)

# Calculate accuracy
accuracy = accuracy_score(test_labels, logistic_predictions)
print(f"Accuracy: {accuracy:.4f}")

# Test your results with the sample tweets below
sample_tweets = [
    "Absolutely loving the new update! Everything runs so smoothly and efficiently now. Great job! 👍",
    "Had an amazing time at the beach today with friends. The weather was perfect! ☀️ #blessed",
    "Extremely disappointed with the service at the restaurant tonight. Waited over an hour and still got the order wrong. 😡",
    "Feeling really let down by the season finale. It was so rushed and left too many unanswered questions. 😞 #TVShow",
    "My phone keeps crashing after the latest update. So frustrating dealing with these glitches! 😠",
]

# Function to predict sentiment of a new tweet
def predict_sentiment(new_text):
    new_vector = vectorizer.transform([new_text])
    pred = logistic_classifier.predict(new_vector)
    return pred[0]

# Test the function
for sentence in sample_tweets:
    print(f"The sentiment predicted by the model is: {predict_sentiment(sentence)}")
