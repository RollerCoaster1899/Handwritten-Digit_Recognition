# Handwritten-Digit_Recognition

1. Importing the necessary libraries:
```python
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
```
The code begins by importing the necessary libraries. `tensorflow` is imported as `tf` for deep learning functionalities. `train_test_split` from `sklearn.model_selection` is used to split the data into training, validation, and testing sets. `accuracy_score` from `sklearn.metrics` is used to calculate the accuracy. `numpy` is imported as `np` for array manipulation. `matplotlib.pyplot` is used for visualization.

2. Loading and preprocessing the MNIST dataset:
```python
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
```
The MNIST dataset is loaded using `tf.keras.datasets.mnist`. The dataset is split into training and testing sets. The pixel values of the images are normalized to the range [0, 1] by dividing by 255.0. The dimensions of the input data are adjusted to match the expected shape of the CNN model.

3. Splitting the data into training, validation, and testing sets:
```python
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
```
The training set is further split into training and validation sets using `train_test_split`. The validation set will be used for monitoring the model's performance during training.

4. Defining the CNN model architecture:
```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```
The model architecture is defined using `tf.keras.Sequential`. It consists of two convolutional layers with ReLU activation followed by max pooling, a flatten layer to convert the 2D feature maps to a 1D vector, a dense layer with ReLU activation, and a final dense layer with softmax activation for multi-class classification. The input shape of the first layer is specified as `(28, 28, 1)`.

5. Compiling and training the model:
```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=128)
```
The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the metric. The model is trained on the training data with validation data used for monitoring. The number of epochs and batch size are specified.

6. Evaluating the model on the test set:
```python
_, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")
```
The trained model is evaluated on the test set using `evaluate()`. The test accuracy is printed.

7. Making predictions and calculating accuracy:
```python
y_pred = np.argmax(model.predict(X_test), axis=-1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
```
Predictions are made on the test set using `model.predict()`, and the class labels with the highest probability are extracted using `np.argmax()`. The accuracy is calculated by comparing the predicted labels with the true labels using `accuracy_score`.

8. Displaying a sample of test images with predictions:
```python
sample_images = X_test[:10]
sample_labels = y_pred[:10]

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, (image, label) in enumerate(zip(sample_images, sample_labels)):
    ax = axes[i // 5, i % 5]
    ax.imshow(np.squeeze(image), cmap=plt.cm.gray_r)
    ax.axis('off')
    ax.set_title(f"Predicted: {label}")
plt.show()
```
A subset of the test images and their corresponding predicted labels are selected. A subplot of 2 rows and 5 columns is created using `plt.subplots()`. Each image is displayed using `imshow()`, and the predicted label is shown as the title of the subplot. The images are visualized using `plt.show()`.

This code uses a CNN model for handwritten digit recognition, performs training with validation, evaluates the model on the test set, calculates accuracy, and displays a sample of test images with their predicted labels. Make sure you have TensorFlow and other required dependencies installed to run this code successfully.
