ROC和AUC
ROC(recursive operating characteristic)和AUC(Area under curve)被定义为ROC曲线的面积


model -> instantiate -> set the hyperparameter-> fit->predict/transform -> accuracy_score
1. LinearRegression 
```python
from sklearn.linear_model import LinearRegression
model=Linearegression(fit_intercept=True)
model.fit(X,y)
```
2. GaussianNB
```python

from  sklearn.cross_validation import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(X_iris,y_iris,random_state=1)

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
# //instantiate model,it is fast and has no hyper-parameters to choose


model.fit(Xtrain, ytrain)
y_model=model.predict(Xtest)



from sklearn.metrices import accuracy_score
accuracy_score(ytest,y_model)

```
3. unsupervised learning example:iris dimensionality 
Often dimensionality reduction is used as an aid to visualizing data.
```python
from sklearn.decomposition import PCA 
model =PCA(n_components=2 )
model.fit(X_iris)
X_2d=model.transform(X_iris)

iris['PCA1']=X_2d[:,0]
iris['PCA2']=X_2d[:,1]
sns.lmplot("PCA1","PCA2",hue="species",data=iris ,fit_reg=False)
```
4. Unsupervised learning:iris clustering
gaussian mixture model :
```python
from sklearn.mixture import GMM
from sklearn.cross_validation import train_test_split
model=GMM(n_components=3,covariance_type='full')
model.fit(X_iris)
y_gmm=model.predict(X_iris )
iris['cluster']=y_gmm

sns.lmplot('PCA1','PCA2',data=iris,hue='species',col='cluster',fit_reg=False)
```
5. Dimensionality reduction:manifold learning 
```python
from sklearn.manifold import Isomap
iso=Isomap(n_components=2)
iso.fit(digits.data)
data_proj=iso.transform(digits.data)
# reduce the dimensionality to 2-dimension in an unsupervised method
```

6. confusion matrix in classification
```python
from sklearn.metrics import confusion_matrix 

mat=confusion_matrix(ytest,y_model)
sns.heatmap(mat,square=True,annot=True,cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')
```
##supervised learning 
1. choose a class of model 
2. Choose model hyperparameter
3. Fit the model to the training data
4. Use the model to predict labels for new data 
### cross-validation 
```python
from sklearn.cross_validation import cross_val_score 
cross_val_score(model,X,y,cv=5)
```
##Selecting the Best model
1. validation_curve
```python
from sklearn.learning_curve import validation_curve

```
2. learning_curve
## Feature Engineering
1.  categorical features 
One-hot coding 
```python

from sklearn.feature_selection import DictVectorizer
vec=DictVectorizer(sparse=True,dtype=int)
vec.transform(data)

```
2. Text features
* word counts
```python
from sklearn.feature_extraction.text import CountVectorizer
vec=CountVectorizer()
X=vec.fit_transform()

```
TF-IDF(term frequency-inverse document frequency),which weights the word counts by a measure of how often in the documents 
```python 
from sklearn.feature_extraction.text import TfidVectorizer
vec=TfidVectorizer()
X=vec.transform(sample)
pd.DataFrame(X.toArray,columns=vec.get_feature_names())


```
3. Derived features 
We saw that we could convert a linear regression into a polynomial regression not by changing the model, but by transforming the input! This is sometimes known as basis function regression
* imputation of missing data
```python
from sklearn.preprocessing import   Imputer
imp=Imputer(strategy='mean')
X1=imp.fit_transform(X)

```
* Feature pipelines
- impute missing values using the mean
- transform features to quafratic
- fit a linear regression
```python
from sklearn.pipeline import make_pipeline
model=make_pipeline(Imputer(strategy='mean'), PolynomialFeatures(degree=2),LinearRegression()
model.fit(X,y)
```
## In depth 
###Naive Bayes classification

        when to use naive bayes
        1. they are extremely fast for both training and prediction
        2. they provide straightforward probabilistic prediction 
        3. they are often very easily interpretable
        4. they have very few (if any) tunable parameters
Performs well in the following situation
        
        1.when the bayes assumptions actually match the data(very rare in practice)
        2. for very well-separated categories,when model complexity is less important
        3. for very high-dimensional data,when model complexity is less important
As the dimension increases, it is much less likely for any two points to be found close together(after all ,they must be close in every single dimension to be close overall)
###Linear regression


##SVM
pros and cons
pros:
* Their dependence on relatively few support vectors means that they are very compact models, and take up very little memory.
* Once the model is trained, the prediction phase is very fast.
* Because they are affected only by points near the margin, they work well with
high-dimensional data—even data with more dimensions than samples, which is
a challenging regime for other algorithms.
* Their integration with kernel methods makes them very versatile, able to adapt to
many types of data.

cons:
* The scaling with the number of samples N is O(N^3) at worst, or O(N^2) for efficient implementations. For large numbers of training samples, this computational cost can be prohibitive.
* The results are strongly dependent on a suitable choice for the softening parame‐ter C . This must be carefully chosen via cross-validation, which can be expensive as datasets grow in size.
* The results do not have a direct probabilistic interpretation. This can be estima‐ted via an internal cross-validation (see the probability parameter of SVC ), but this extra estimation is costly.
## Decision trees and Random Forests
1. This notion-that multiple over-fitting estimators can be combined to reduce the effect of this over-fitting is what underlies an ensemble method called *bagging*.Bagging makes use of an ensemble(a grab perhaps) of parallel estimators, each of which over-fits the data,and averages the results to find a better classification. An ensemble of randomized decision trees is known as a random forrest.


2. random forests can also be made to work in the case of regression 
```python
from sklearn.ensemble import RandomForestregressor
forest= RandomForestRegressor(200)

```

3. summary
* Both training and prediction are very fast, because of the simplicity of the under‐lying decision trees. In addition, both tasks can be straightforwardly parallelized,because the individual trees are entirely independent entities.
* The multiple trees allow for a probabilistic classification: a majority vote among estimators gives an estimate of the probability (accessed in Scikit-Learn with the predict_proba() method).
* The nonparametric model is extremely flexible, and can thus perform well on tasks that are under-fit by other estimators.


Disadvantages:A primary disadvantage of random forests is that the results are not easily interpretable;that is ,to draw conclusions about the meaning of the classification model is difficult


## Principal Component Analysis

This transformation from data  axes to principal axes is as an *affine transformation *,which basically means it is composed of a translation,rotation,and uniform scaling 


1. PCA as dimensionality reduction 
Using PCA for dimensionality reduction involves zeroing out one or more of the smallest principal components, resulting in a lower-dimensional projection of the data that preserves the maximal data variance.

what a PCA dimensionality reduction means :the information along the least important principal  axis or axes is removed ,leaving only the component of the data with the highest variance.

2. PCA as Noise Filtering 
PCA can also be used as a filtering approach for noisy data.The idea is this :any components with variance much larger than the effect of the noise should be relatively unaffected by the noise 
3. PCA's summary 

Given any high-dimensional dataset, I tend to start with PCA in
order to visualize the relationship between points (as we did with the digits), to
understand the main variance in the data (as we did with the eigenfaces), and to
understand the intrinsic dimensionality (by plotting the explained variance ratio).

PCA’s main weakness is that it tends to be highly affected by outliers in the data.
sklearn contains a couple interesting variants on PCA,including *RandomsizedPCA,and SparsePCA*both in the *sklearn.decomposition* submodule.*RandomizedPCA* , which we saw earlier, uses a non-deterministic method to quickly approximate the first few principal components in very high-dimensional data, while *SparsePCA* introduces a regularization term
that serves to enforce sparsity of the components
## Manifold learning 
流形学习的前提是有一种假设，即某些高维数据，实际是一种低维的流形结构嵌入在高维空间中。流形学习的目的是将其映射回低维空间中，揭示其本质。
manifold methods:MDS(multidimensional scaling),locally linear embedding(LLE),and isometric mapping(Isomap)
1. MDS(multidimensional scaling):even a distance matrix between points, it recovers a D -dimensional coordinate representation
of the data. Let’s see how it works for our distance matrix, using the 
dissimilarity to specify that we are passing a distance matrix

**Manifold learning estimator:given high-dimensional embedded data,it seeks a low-dimensional representation of the data that preserves certain relationship within the data.** In the case of MDS, the quantity preserved is the distance between every pair of points.

2. Nonlinear Embedding: Where MDS Fails
LLE:rather than preserving all distances,it instead tries to preservr only the distances between *neighboring points*:in this case,the nearest 100 neighbors of each point.

3. Comparison with PCA
* In manifold learning, there is no good framework for handling missing data. In contrast, there are straightforward iterative approaches for missing data in PCA.
* In manifold learning, the presence of noise in the data can “short-circuit” the manifold and drastically change the embedding. In contrast, PCA naturally filters noise from the most important components.
* The manifold embedding result is generally highly dependent on the number of neighbors chosen, and there is generally no solid quantitative way to choose an optimal number of neighbors. In contrast, PCA does not involve such a choice.
* In manifold learning, the globally optimal number of output dimensions is difficult to determine. In contrast, PCA lets you find the output dimension based on the explained variance.
* In manifold learning, the meaning of the embedded dimensions is not always clear. In PCA, the principal components have a very clear meaning.
* In manifold learning the computational expense of manifold methods scales as O[N^2] or O[N^3]. For PCA, there exist randomized approaches that are generally much faster (though see the megaman package for some more scalable implementations of manifold learning).
4. Recommendations
* For toy problems such as the S-curve we saw before, locally linear embedding (LLE) and its variants (especially modified LLE), perform very well. This is implemented in sklearn.manifold.LocallyLinearEmbedding .
* For high-dimensional data from real-world sources, LLE often produces poor results, and isometric mapping (Isomap) seems to generally lead to more meaningful embedding. This is implemented in sklearn.manifold.Isomap .
* For data that is highly clustered, t-distributed stochastic neighbor embedding (t-SNE) seems to work very well, though can be very slow compared to other methods. This is implemented in sklearn.manifold.TSNE .
## K-means Clustering 
k-means accomplishes clustering using a simple conception of what the optimal clustering looks like:
* The 'cluster center' is the arithmetic mean of all the points belonging to the cluster
* Each point is closer to its own cluster center than to other cluster centers
k-means algorithm:Expectation-Maximization:
1. Guess some cluster centers 
2. Repeat until converged 
        a. E-Step:assign points to the nearest cluster center 
        b. M-Step: set the cluster centers to the mean


**Warning of EM algorithm**:
1. The globally optimal result may not be achieved
2. The number of clusters must be selected beforehand
3. *k-means* is limited to linear cluster boundaries
The fundamental model assumptions of k-means (points will be closer to their own cluster center than to others) means that the algorithm will often be ineffective if the clusters have complicated geometries.


**One version of this kernelized k-means is implemented is the SpecturalClustering estimator**.It uses the graph of nearest neighbors to compute a higher-dimensional representation of the data,and then assign labels using a *k-means* algorithm.**The kernelized k-means is able to find the more compicated nonlinar boundaries between clusters**
4. *k-means* can be slow for large number of samples. *t-SNE*is a nonlinear embedding algorithm that is particularly adept at preserving points within clusters.


##Gaussian Mixture Models
In particular, the non-probabilistic nature of k-means and its use of simple distance-from-cluster-center to assign cluster membership leads to poor performance for many real-world situations. In this section we will take a look at Gaussian mixture models, which can be viewed as an extension of the ideas behind k-means, but can also be a powerful tool for estimation beyond simple clustering.

So as to account for non-circular clusters,It turn out these are two essential components of a different type of clustering model,Gaussian mixture models

1. Generalizing E-M:Gaussian Mixture Models 
A Gaussian mixture model(GMM) attempts to find a mixture of multidimensional Gaussian probability distributions that best model any input model
```python
from sklearn.mixture import GMM
gmm=GMM(n_components=4).fit(X)
labels=gmm.predict(X)

```
2. GMM as Density Estimation
Though GMM is often categorized as a clustering algorithm,fundamentally it is an algorithm for *density estimation*.This is to say, the result of a GMM fit to some data is technically not a clustering model,but a generative probabilistic model describing the distribution of the data


        A generative model is inherently a probability distribution for the dataset.

        Notice this choice of number of components measures how well GMM works as a density estimator,not how well it works as a clustering algorithm
## Kernel Density Estimation
*GMM*：a kind of hybrid between a clustering estimator and a density estimator
*KDE*:an algorithm that takes the mixture-of-Gaussian idea to its logical extreme:it used a mixture consisting of one Gaussian component per point,resulting in an essentially non-pararmetric estimator of density.


        less variance:changes much less in response to differences in sampling 
```python
from sklearn.neighbors import KernelDensity
kde=KernelDensity(bandwidth=1.0,kernel='gaussian')
kde.fit(x[:,None])

```
1. selecting the bandwidth via cross-validation
The choice of bandwidth within KDE is extremely important to finding a suitable
density estimate, and is the knob that controls the bias–variance trade-off in the estimate of density: too narrow a bandwidth leads to a high-variance estimate (i.e., over‐
fitting), where the presence or absence of a single point makes a large difference. Too
wide a bandwidth leads to a high-bias estimate (i.e., underfitting) where the structure
in the data is washed out by the wide kernel.
```python
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import LeaveOneOut


bandwidths =10**np.linspace(-1,1,100)
grid=GridSearchCV(KernelDensity(kernel='gaussian'),{
        'bandwidth':bandwidths
},cv=LeaveOneOut(len(x)))

grid.fit(x[:,None])

grid.best_params_

```



## The general  approach for generative classification is this:
1. Split the training data by label 
2. For each set, fit a KDE to obtain a generative model of the data.This allows you for any observation $x$ and label $y$ to compute a likelihood $P(x|y)$
3. From the number of examples of each class in the training set,compute the class priori $P(y)$
4. For an unknown point $x$,the posterior probability for each class is $P(y|x)\propto{P(x|y)P(y)}$.The class that maximizes this posterior is the label assigned to the point



## The HOG faces detector
1. Obtain a set of image thumbnails of faces to constitute "positive" training samples
2. Obtain a set of image thumbnails of non-faces to constitute "negative" training samples
3. Extract HOG features from these training samples
4. Train a linear SVM classifier on these samples 
5. For an "unknown" image ,pass a sliding window across the image,using the model to evaluate whether that window contains a face or not 
6. If detections overlap,combine them into a single window
 
