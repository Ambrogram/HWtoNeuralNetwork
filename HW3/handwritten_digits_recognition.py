# Import necessary libraries
import os  # Provides functions to interact with the operating system
import cv2  # OpenCV library for image processing
import numpy as np  # NumPy library for numerical operations
import tensorflow as tf  # TensorFlow library for machine learning and neural networks
import matplotlib.pyplot as plt  # Matplotlib's pyplot for plotting graphs and images

# Print a welcome message
print("Welcome to the NeuralNine (c) Handwritten Digits Recognition v0.1")

# Decide whether to train a new model or load an existing one
train_new_model = True  # Boolean flag to control the workflow

# Check if training a new model is required
if train_new_model:
    # Load the MNIST dataset (a dataset of handwritten digits)
    mnist = tf.keras.datasets.mnist  # Access the MNIST dataset from TensorFlow's dataset collection
    (X_train, y_train), (X_test, y_test) = mnist.load_data()  # Load the dataset and split into training and testing sets

    # Normalize the data: scale pixel values to a range of 0 to 1
    X_train = tf.keras.utils.normalize(X_train, axis=1)  # Normalize training images
    X_test = tf.keras.utils.normalize(X_test, axis=1)  # Normalize testing images

    # Create a Sequential neural network model
    model = tf.keras.models.Sequential()  # Sequential model, linear stack of layers
    model.add(tf.keras.layers.Flatten())  # Flatten layer to convert 2D pixel map to 1D array
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))  # First dense layer with 128 neurons, ReLU activation
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))  # Second dense layer with 128 neurons, ReLU activation
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))  # Output layer with 10 neurons (digits 0-9), softmax activation

    # Compile the model with optimizer, loss function, and metrics
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model with the training data
    model.fit(X_train, y_train, epochs=3)  # Fit the model for 3 epochs

    # Evaluate the model using the test data
    val_loss, val_acc = model.evaluate(X_test, y_test)  # Evaluate and get loss and accuracy
    print(val_loss)  # Print validation loss
    print(val_acc)  # Print validation accuracy

    # Save the trained model
    model.save('handwritten_digits.model')  # Save the model to a file
else:
    # Load an existing model
    model = tf.keras.models.load_model('handwritten_digits.model')  # Load the model from the file

# Predicting digits from custom images
image_number = 1  # Start with the first image
while os.path.isfile('digits/digit{}.png'.format(image_number)):  # Check if the image file exists
    try:
        # Read and process the image
        img = my_cv2_test.imread('digits/digit{}.png'.format(image_number))[:,:,0]  # Read the image, take only the first color channel
        img = np.invert(np.array([img]))  # Invert the colors and convert to NumPy array
        prediction = model.predict(img)  # Predict the digit using the model
        print("The number is probably a {}".format(np.argmax(prediction)))  # Print the predicted digit
        plt.imshow(img[0], cmap=plt.cm.binary)  # Show the image
        plt.show()  # Display the plot
        image_number += 1  # Move to the next image
    except:
        # If an error occurs during image processing
        print("Error reading image! Proceeding with next image...")  # Print error message
        image_number += 1  # Move to the next image
