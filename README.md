<H3>Name:Rithika K</H3>
<H3>Register no:212224230230</H3>
<H3>Date:05/08/2026</H3>
<H3>Experiment No. 2 </H3>
# Implementation of Perceptron for Binary Classification

# AIM:

To implement a perceptron for classification using Python<BR>

# EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

# RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.<BR>
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.<BR>
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.<BR>
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron. <BR>
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’
f(x)=w.x+b
 <BR>
A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

<img width="283" alt="image" src="https://github.com/Lavanyajoyce/Ex-2--NN/assets/112920679/c6d2bd42-3ec1-42c1-8662-899fa450f483">


Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.<BR>


# ALGORITHM:
STEP 1: Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Plot the data to verify the linear separable dataset and consider only two classes<BR>
STEP 4:Convert the data set to scale the data to uniform range by using Feature scaling<BR>
STEP 4:Split the dataset for training and testing<BR>
STEP 5:Define the input vector ‘X’ from the training dataset<BR>
STEP 6:Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2<BR>
STEP 7:Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
STEP 8:Assign the learning rate<BR>
STEP 9:For ‘N ‘ iterations ,do the following:<BR>
        v(i) = w(i)*x(i)<BR>
         
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)<BR>
STEP 10:Plot the error for each iteration <BR>
STEP 11:Print the accuracy<BR>
# PROGRAM:
```
   import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris


# ---------------------------------------------------
# Perceptron Class
# ---------------------------------------------------

class Perceptron:

    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0
        self.errors = []

    # Sigmoid activation function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    # Training function
    def fit(self, X, y):

        # Initialize weights to zero
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        self.errors = []

        for i in range(self.n_iterations):

            errors = 0

            for xi, target in zip(X, y):

                # Calculate weighted sum
                v = np.dot(xi, self.weights) + self.bias

                # Sigmoid activation
                output = self.sigmoid(v)

                # Convert output to +1 or -1
                prediction = 1 if output >= 0.5 else -1

                # Calculate weight update
                update = self.learning_rate * (target - prediction)

                # Update weights
                self.weights += update * xi

                # Update bias
                self.bias += update

                # Count errors
                if update != 0:
                    errors += 1

            self.errors.append(errors)

        return self

    # Prediction function
    def predict(self, X):

        v = np.dot(X, self.weights) + self.bias

        output = self.sigmoid(v)

        return np.where(output >= 0.5, 1, -1)


# ---------------------------------------------------
# Main Program - Load Iris Dataset
# ---------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target


# ---------------------------------------------------
# Consider only two classes
# Iris-setosa = 0
# Iris-versicolor = 1
# ---------------------------------------------------

mask = y < 2

X = X[mask]
y = y[mask]


# ---------------------------------------------------
# Map labels to binary values +1 and -1
# ---------------------------------------------------

y = np.where(y == 0, 1, -1)


print("Dataset loaded successfully")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])


# ---------------------------------------------------
# Standardization of input features
# ---------------------------------------------------

X = (X - X.mean(axis=0)) / X.std(axis=0)

print("\nData standardized successfully")


# ---------------------------------------------------
# Split the data into training and testing
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ---------------------------------------------------
# Create and train Perceptron
# Learning rate = 0.01
# ---------------------------------------------------

model = Perceptron(
    learning_rate=0.01,
    n_iterations=100
)

model.fit(X_train, y_train)

print("\nModel training completed")


# ---------------------------------------------------
# Predict test data
# ---------------------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------------------
# Calculate accuracy
# ---------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nActual values:")
print(y_test)

print("\nPredicted values:")
print(y_pred)

print("\nAccuracy:", accuracy * 100, "%")


# ---------------------------------------------------
# Plot number of errors during each iteration
# ---------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(model.errors) + 1),
    model.errors,
    marker='o'
)

plt.xlabel("Iteration")
plt.ylabel("Number of Errors")
plt.title("Perceptron Training Error")

plt.grid(True)
plt.show()
```

# OUTPUT:
<img width="1027" height="603" alt="image" src="https://github.com/user-attachments/assets/78a22691-cb94-4fc7-9da5-e3c1b92a5fca" />
<img width="789" height="221" alt="image" src="https://github.com/user-attachments/assets/03834303-41cf-472b-b0ea-e988257df023" />


# RESULT:
 Thus, a single layer perceptron model is implemented using python to classify Iris data set.

 
