# Training 
Note that some regression algorithms can be used for classification as well, and vice
versa. For example, Logistic Regression is commonly used for classification, as it can
output a value that corresponds to the probability of belonging to a given class
##supervised learning:
* k-Nearest Neighbors
* Linear Regression
* Logistic Regression
*  Support Vector Machines (SVMs)
* Decision Trees and Random Forests
* Neural networks 2

        Some neural network architectures can be unsupervised, such as autoencoders and restricted Boltzmann
        machines. They can also be semisupervised, such as in deep belief networks and unsupervised pretraining.

## unspervised learning:
1. clustering:
- k-maens
- hierarchical cluster analysis(HCA)
- expectation maximization
2.  visualization and dimensionality reduction
- principal  concept analysis(PCA)
- kernel PCA 
- locally-linear embedding(LLE)
- t-distributed stochastic neighbor embedding (t-SNE)
3. association rule learning:
- APriori 
- eclat

One important unsupervised task is anomaly detection.Another common unsupervised task is association rule learning, in which the goal is to dig into large amounts of data and discover interesting relations between
attributes.

## semi-supervised learning：
Most semisupervised learning algorithms are combinations of unsupervised and supervised algorithms. For example, deep belief networks (DBNs) are based on unsu‐pervised components called restricted Boltzmann machines (RBMs) stacked on top of one another. RBMs are trained sequentially in an unsupervised manner, and then the whole system is fine-tuned using supervised learning techniques.

## Batch learining:
In batch learning, the system is incapable of learning incrementally: it must be trained using all the available data. This will generally take a lot of time and computing resources, so it is typically done offline. First the system is trained, and then it is launched into production and runs without learning anymore; it just applies what it has learned. This is called offline learning.

## online learning:
Online learning is great for systems that receive data as a continuous flow (e.g., stockprices) and need to adapt to change rapidly or autonomously.
In online learning, you train the system incrementally by feeding it data instances  sequentially, either individually or by small groups called mini-batches.

    One important parameter of online learning systems is how fast they should adapt to
    changing data: this is called the learning rate.

## Instanced-based versus model-based learning:

    One more way to categorize Machine Learning systems is by how they generalize.
1. instanced-based learning
the system learns the examples by heart, then generalizes to new cases using a similarity measure
2. model-based learning 
Another way to generalize from a set of examples is to build a model of these exam‐ples, then use that model to make predictions. This is called model-based learning

3. utility function(效用函数)
measures how good model is
4. cost function:
measure how bad model is 
5. sampling bias:
sampling method is flawed

## overfitting 


Overfitting happens when the model is too complex relative to the amount and noisiness of the training data. The possible solutions are:
1. To simplify the model by selecting one with fewer parameters
(e.g., a linear model rather than a high-degree polynomial model), by reducing the number of attributes in the training data or by constraining the model
2. To gather more training data
3. To reduce the noise in the training data (e.g., fix data errors and remove outliers)

## regularizaion 
Constraining a model to make it simpler and reduce the risk of overfitting is called regularization.

## hyperparameter 
The amount of regularization to apply during learning can be controlled by a hyper‐parameter.

A hyperparameter is a parameter of a learning algorithm (not of the model).As such,it is not attected by the learning algorithm itself; it must be set prior to training and reamins consant during training.


## underfitting the training data
It occurs when your model is too simple to learn the underlying structure of the data.

The main options to fix this problem are:
* Selecting a more powerful model, with more parameters
* Feeding better features to the learning algorithm (feature engineering)
* Reducing the constraints on the model (e.g., reducing the regularization hyper‐
parameter)


# Testing and validating 
split data into training set and test set 
### generalization error
The error rate on new cases 
## validation  
You train multiple models with various hyper-parameters using the training set, you select the model and hyperparameters that perform best on the validation set,and when you’re happy with your model you run a single final test against the test set to get an estimate of the generalization error.