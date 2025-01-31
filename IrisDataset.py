"""
Classification of the iris dataset using neural networks.

Explanation for parameters:
  The parenthesis of the hidden_layer_sizes are options where the first number
  is the number of neurons in the first layer and the next number is the number
  of neurons in the second layer and so on.

  In this example, we are testing different variations of the number of neurons in each layer and the number of layers.
  For example if parameters was equal to {"hidden_layer_sizes": [(1,),(4,),(2,),(2,3)]}, 
  the first option would be just one neuron in one layer and the second option is four neurons in one layer. 
  The third option contains two neurons in one layer and fourth option has two neurons in the first layer and three in the second.
"""

from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Step One: Data
X,y = load_iris(return_X_y = True) # loading iris dataset data. return_X_y makes sure that the data comes as a NumPy array instead of a Scikit Learn bunch.
X_train, X_test, y_train, y_test = train_test_split(X,y)

# Step Two: Algorithm
mlpalgo = MLPClassifier(max_iter = 5000)
parameters = {"hidden_layer_sizes": [(1,),(4,),(2,),(2,3)]} # trying different options for the hidden layer sizes. 
algo = GridSearchCV(mlpalgo,parameters,verbose=4) # loops over the options and verbose shows how much you want it to show about each option

# Step Three: Training
algo.fit(X_train, y_train)

# Step Four: Testing
y_pred = algo.predict(X_test) # y_pred uses the best option for hidden layer sizes

# Step Five: Analysis
# Training Analysis
print(algo.best_estimator_) # Best hidden layer size option
print(algo.best_score_) 

# Testing Analysis
score = accuracy_score(y_test, y_pred)
print(score)
plt.scatter(X_train[:,2],X_train[:,3], c = y_train, s = 18, label = "Training Data")
plt.scatter(X_test[:,2],X_test[:,3], c = y_pred, marker = "*", s = 85, label = "Testing Data")
plt.title("Neural Network Classification on Iris Flower Dataset")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()

