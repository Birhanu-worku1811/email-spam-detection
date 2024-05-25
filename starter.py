import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

# Read the dataset
emails = pd.read_csv("emails.csv")


# Preprocess the emails
def process_email(text):
    text = text.lower()
    return list(set(text.split()))


emails["words"] = emails["text"].apply(process_email)


# Find Priors
def calculate_priors(emails):
    total_emails = len(emails)
    spam_emails = len(emails[emails["spam"] == 1])
    ham_emails = len(emails[emails["spam"] == 0])
    prior_spam = spam_emails / total_emails
    prior_ham = ham_emails / total_emails
    return prior_spam, prior_ham


prior_spam, prior_ham = calculate_priors(emails)


# Find Posteriors with Bayes' Theorem
def calculate_likelihoods(emails):
    spam_emails = emails[emails["spam"] == 1]
    ham_emails = emails[emails["spam"] == 0]
    spam_words = {}
    ham_words = {}

    for email in spam_emails["words"]:
        for word in email:
            if word in spam_words:
                spam_words[word] += 1
            else:
                spam_words[word] = 1

    for email in ham_emails["words"]:
        for word in email:
            if word in ham_words:
                ham_words[word] += 1
            else:
                ham_words[word] = 1

    return spam_words, ham_words


spam_words, ham_words = calculate_likelihoods(emails)


def calculate_posteriors(word, spam_words, ham_words, total_spam_words, total_ham_words, vocab_size):
    likelihood_spam = (spam_words.get(word, 0) + 1) / (total_spam_words + vocab_size)
    likelihood_ham = (ham_words.get(word, 0) + 1) / (total_ham_words + vocab_size)
    return likelihood_spam, likelihood_ham


total_spam_words = sum(spam_words.values())
total_ham_words = sum(ham_words.values())
vocab_size = len(set(spam_words.keys()).union(set(ham_words.keys())))


# Implementing the naive Bayes algorithm
def predict_naive_bayes(email, spam_words, ham_words, prior_spam, prior_ham, total_spam_words, total_ham_words,
                        vocab_size):
    email_words = process_email(email)
    spam_score = np.log(prior_spam)
    ham_score = np.log(prior_ham)

    for word in email_words:
        likelihood_spam, likelihood_ham = calculate_posteriors(word, spam_words, ham_words, total_spam_words,
                                                               total_ham_words, vocab_size)
        spam_score += np.log(likelihood_spam)
        ham_score += np.log(likelihood_ham)

    if spam_score > ham_score:
        return "spam"
    else:
        return "not spam"


# Flask API
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    email = data.get("email", "")
    result = predict_naive_bayes(email, spam_words, ham_words, prior_spam, prior_ham, total_spam_words, total_ham_words,
                                 vocab_size)
    return jsonify({"result": result})


if __name__ == '__main__':
    app.run(debug=True)
