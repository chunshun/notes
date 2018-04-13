

A scalar can be defined as a rank-0 tensor,a vector as a rank-1 tensor,a matrix as a rank-2 tensor,and matrices stacked as rank-3 tensor   

matrix:*batch-size\*2\*3*

```python 
x=tf.placeholder(dtype=tf.float32,shape=(None),name='x')

```
`tf.reshape(tensor,shape=(-1,))` can be used to flatten a tensor


## Training neural networks efficiently with high-level Tensor-flow APIs

`tf.layers` and `tf.contrib.keras` 

## Tensor-flow ranks and tensors
Tensors are a generalizable mathematical notation for multidimensional arrays holding data values,where the dimensionality of a tensor is typically referred to as its **rank**  

As we can see,the rank of the `t1` tensor is 0 since it is just a scalar(corresponding to the [] shape).The rank of the `t2` vector is 1,and since it has four elements,its shape is the one-element tuple `(4,)` .Lastly,the shape of the 2*2 matrix `t3` is 2;  thus ,its corresponding shape is given by the `(2*2)` tuple


### Defining variables
```python 
tf.Variable(<initial-value>,name="variable-name")
#or
tf.get_variable(name,...)
```

Note that `tf.Variable` does not have an explicit way to determine `shape` and `dtype`; the shape and type are set to those of the initial values.It stores the parameter of a model that can be updated during training.**When we define a variable,we need to initialize it with a tensor of values**  

The second option `tf.get_variable` can be used  to **reuse** an existing variable with a given name(if the name exists in the graph) or create a new one if the name does not exist.Furthermore,`tf.get_variable` provides an explicit way to set `shape` and `dtype`; these parameters are only required when creating a new variable, not reusing existing ones.   

The advantage of `tf.get_variable` over `tf.Variable` is twofold:`tf.get_variable` allows us to reuse existing variables it already uses  the popular Xavier/Glorot initialization scheme by default.  


*The initialization process refers to allocating memory for the associated their initial values*


```python 
import tensorflow as tf

g=tf.Graph()


with g.as_default():
    w1=tf.Variable(1,name='w1')
    init_op=tf.global_variables_initializer()



with tf.Session(graph=g)  as sess:
    sess.run(init_op)


```
## Variable scope
With variable scopes,we can organize the variables into separate subparts.When we create a variable scope, the name of operations and tensors that are created within that scope are prefixed with   that scope,and those scopes can further be nested
```python
import tensorflow as tf

g=tf.Graph()


with g.as_default():
    with tf.variable_scope('net_A'):
        with tf.variable_scope('layer_1'):
            w1=tf.Variable(tf.random_normal(shape=(10,4),name='weights'))
        with tf.variable_scope('layer_2'):
            w2=tf.Variable(tf.random_normal(shape=(20,10),name='weights'))
    with tf.variable_scope('net_B'):
        with tf.variable_scope('layer_1'):
            w3=tf.Variable(tf.random_normal(shape=(10,4),name='weights'))



```

### Building a regression model
```python
import tensorflow as tf

g=tf.Graph()


with g.as_default():
    tf.set_random_seed(123)
    ##placeholders
    tf_x=tf.placeholder(shape=None,dtype=tf.float32,name='tf_x')
    tf_y=tf.placeholder(shape=None,dtype=tf.float32,name='ty_y')
    ## define the variable(model parameters)
    weight=tf.Variable(
        tf.random_normal(
            shape=(1,1)
            stddev=0.25,

        ),name='weight'
    )
    bias=tf.Variable(0.0,name='bias')
    # build the model
    y_hat=tf.add(weight*tf_x,bias,name='y_hat')
    # compute the cost
    cost=tf.reduce_mean(tf.square(tf_y-y_hat),name='cost')
    
    # train the model
    opti=tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op=opti.minimize(cost,name='train_op')
    


with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    # train the model for n_epochs
    for e in range(n_epochs):
        c,_=see.run([cost,train_op],feed_dict={
            tf_x:x_train,
            tf_y:y_train
        })
        training_costs.append(c)
        if not e%50:
            print('Epoch %4d: %.4f'%(e,c))


```
## Saving and Using a model in Tensorflow
1. Rebuild the graph that has the same nodes and names as the saved model
2. Restore the saved variables in a new `tf.Session` environment



## The control mechanics of Tensorflow
```python

import tensorflow as tf


res=tf.cond(
    tf_x<tf_y,
        lambda : tf.add(tf_x,tf_y, name='result_add'),

        lambda : tf.subtract(tf_x,tf_y,name='result_sub')

)
``` 

In addition to `tf.cond`,Tensorflow offers serval other control flows operators,such as `tf.case` and `tf.while_loop`.

```python
f1=lambda :tf.constant(1)
f2=lambda :tf.constant(0)


result = tf.case([(tf.less(x,y),f1)],default=f2)



## while_loop

i=tf.constant(0)
threshold=100
c=lambda i:tf.less(i,100)
b=lambda i:tf.add(i,1)
r=tf.while_loop(cond=c, body=b,loop_vars=[i])


```

## CNN 
CNNs will usually perform very well for image-related tasks,and that's largely due to two important ideas:
- **Sparse-connectivity**:A single element in the feature map is connected to only a small patch of pixels.
- **Parameter-sharing**:The same weights are used for different patches of the input image   


   
Typically, CNNs are composed of several **Convolutional(conv)** layers and sub-sampling(also known as **Pooling(p)**) layers that are followed by one or more **Fully Connected(FC)** layers at the end.The fully connected layer is essentially a multilayer perceptron   

Note that sub-sampling layers,commonly known as **pooling layers**,do not have any learnable parameters; for instance, there are no weights or bias units in pooling layers.However, both convolution and fully connected layers have such weights and biases   

