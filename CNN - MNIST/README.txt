Project: Convolutional Neural Network (CNN) for MNIST Digit Classification
Author: Saad
Description:
------------
This mini project builds and trains a Convolutional Neural Network (CNN) to classify handwritten digits (0–9) using the MNIST dataset. The project follows a structured notebook workflow with clear explanations, reshaping, normalization, model building, training, and evaluation.

Dataset:
--------
The MNIST dataset contains:
- 60,000 training images
- 10,000 testing images
- Each image is 28x28 pixels, grayscale (1 channel)

Steps Completed:
----------------

1. Downloaded the MNIST dataset using:
   from keras.datasets import mnist
   (x_train, y_train), (x_test, y_test) = mnist.load_data()

2. Reshaped the images to add a channel dimension:
   (60000, 28, 28) → (60000, 28, 28, 1)
   (10000, 28, 28) → (10000, 28, 28, 1)

3. Normalized pixel values by dividing by 255.0
   This converts pixel values from 0–255 to 0–1 for better training stability.

4. Built a CNN with the following architecture:
   - Conv2D layer: 28 filters, 3x3 kernel, ReLU activation
   - MaxPooling2D layer: 2x2 pool size
   - Flatten layer
   - Dense layer: 128 units, ReLU activation
   - Output Dense layer: 10 units, Softmax activation

5. Compiled the model using:
   - Optimizer: Adam
   - Loss: sparse_categorical_crossentropy
   - Metrics: accuracy

6. Trained the model for 100 epochs using:
   model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

7. Evaluated the model using:
   model.evaluate(x_test, y_test)

Outputs:
--------
The model returns:
- Final Test Loss
- Final Test Accuracy

These values indicate how well the model performs on unseen test images.

Notes:
------
- Reshaping does NOT change the number of images; it only adds the channel dimension.
- Dividing by 255 normalizes pixel values but does NOT change dataset size.
- Training and testing sets are pre-split by MNIST to prevent overfitting.
- Softmax ensures the output layer produces probabilities for digits 0–9.