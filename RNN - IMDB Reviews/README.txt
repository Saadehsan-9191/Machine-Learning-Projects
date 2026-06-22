Project Overview
This mini project builds a Recurrent Neural Network (RNN) using Keras to classify IMDB movie reviews as positive or negative.
The dataset contains 50,000 labeled reviews. The model uses tokenization, padding, embeddings, and an LSTM layer to learn sentiment patterns from text.

Dataset
The dataset used is IMDB Dataset.csv.
It contains two columns:

review : the movie review text

sentiment : "positive" or "negative"

Sentiment labels are converted to numeric values:
positive = 1
negative = 0

Preprocessing Steps

Load the CSV dataset

Remove empty or missing reviews

Encode sentiment labels

Split the dataset into training and testing sets (80 percent train, 20 percent test)

Tokenize the text using Keras Tokenizer

Convert text to sequences of integers

Remove empty sequences

Pad all sequences to a fixed length of 200 words

Model Architecture
The model is a simple RNN with the following layers:

Embedding layer with 128 dimensions

LSTM layer with 128 units

Dropout layer with 30 percent dropout

Dense output layer with sigmoid activation for binary classification

Training Details
Loss function: binary_crossentropy
Optimizer: Adam
Metrics: accuracy
Epochs: 5
Batch size: 128
Validation split: 20 percent of training data

Evaluation
The model is evaluated on the test set using accuracy and loss.
Typical accuracy for this architecture is between 85 and 90 percent depending on preprocessing.

Prediction Function
A helper function is included to classify new reviews.
The function takes a text string, tokenizes it, pads it, and returns either "Positive" or "Negative".

Files Included

IMDB Dataset.csv

notebook.ipynb

README.txt

Summary
This project demonstrates how to clean text data, tokenize and pad sequences, build an RNN model using Keras, train it on movie reviews, and evaluate its performance on unseen data.