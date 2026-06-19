Shakespeare RNN Mini Project
============================

This project trains an LSTM-based language model on Shakespeare text.

Steps Performed:
1. Load and preprocess text (lowercase, split lines)
2. Tokenize text and build vocabulary
3. Convert lines to sequences of integers
4. Create subsequences for next-word prediction
5. Pad sequences to equal length
6. Split into X (inputs) and y (labels)
7. One-hot encode labels
8. Build LSTM model:
   - Embedding(100)
   - LSTM(100)
   - Dropout(0.1)
   - Dense(vocab_size, softmax)
9. Train for 500 epochs using Adam optimizer

Output:
- A trained Keras model capable of predicting the next word in Shakespeare-like text.

Files Needed:
- data.txt (Shakespeare text)

How to Run:
- Place data.txt in the same folder
- Run all notebook cells
- Model will train and return the final trained model