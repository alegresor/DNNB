
# Deep Neural Network Builder (DNNB)

The following sections describe folders and files rooted at `/Power_Systems/dnnb/`.\
Mathematical formulations are followed by example class instantiations. 

## Notation

- $.$ = dot product
- $\text{*}$ = element wise product
- $:$ = all indices along the axis
- $a:b$ = all indices $i$ along the axis such that $a \leq i < b$

## Layers

**Linear**

$z_{j} = x.w_{:,j} + b_j$

- $z$ = output (output_features)
- $x$ = input (input_features)
- $w$ = weight (input_features x output_features)
- $b$ = bias (output_features)
- $j$ = feature index

~~~python
Linear(
    input_features = 188,
    output_features = 1) 
~~~

**Convolution 2D / 3D**

$z_{l,j} = sum(x_{:,js:js+f} \text{*} k_{l,:,:}) + b_l$

- $z$ = output (kernel_filters x (input_features-kernel_features)/s+1)
- $x$ = input (input_filters x input_features)
- $k$ = kernel (kernel_filters x input_filters x kernel_features}
- $b$ = bias (kernel_filters)
- $l$ = filter index
- $j$ (2D) or $(j1,j2)$ (3D) = feature index
- $s$ (2D) or $(s1,s2)$ (3D) = stride
- $f$ (2D) or $(f1,f2)$ (3D) = kernel features

~~~python
Conv_2D(
    input_filters = 1,
    input_features = 188,
    kernel_filters = 2,
    kernel_features = 2,
    stride = 2,
    padding = 0)
~~~

~~~python
Conv_3D(
    input_filters = 1,
    input_features = (28,28),
    kernel_filters = 24,
    kernel_features = (5,5),
    stride = (1,1),
    padding = 0)
~~~

**Graph Convolutional**

$x_{v,j} = a_{v,:}.{r}_{:,j}$\
$z_{v,j} = x_{v,:}.w_{:,j} + b{j}$
    
- $r$ = input (nodes x input_features)
- $a$ = adjacency matrix (nodes x nodes)
- $x$ = input after accounting for adjacency matrix (nodes x input_features)
- $z$ = output (nodes x output_features)
- $w$ = weight (input_features x output_features)
- $b$ = bias (output_features)
- $v$ = node index
- $j$ = feature index

~~~python
GraphConv(
    adjacency_matrix = array([[1,1,0],[1,1,0],[0,0,1]]),
    input_features = 3,
    output_features = 2)
~~~

## Transforms

**Reshape**

$x$ (old_shape) $\rightarrow$ $x$ (new_shape)

~~~python
Reshape(1,188)
~~~

**Max Pooling 2D / 3D**

$z_{l,j} = max(x_{l,js:js+f})$

- $z$ = output (kernel_filters x (input_features-kernel_features)/s+1)
- $x$ = input (input_filters x input_features)
- $l$ = kernel index
- $j$ (2D) or $(j1,j2)$ (3D) = feature index
- $s$ (2D) or $(s1,s2)$ (3D) = stride
- $f$ (2D) or $(f1,f2)$ (3D) = kernel features

~~~python
MaxPooling_2D(
    input_features = 94,
    kernel_features = 2,
    stride = 2)
~~~

~~~python
MaxPooling_3D(
    input_features = (8,8),
    kernel_features = (2,2),
    stride = (2,2))
~~~

**Average Pooling 2D / 3D**

$z_{l,j} = mean(x_{l,js:js+f})$

- $z$ = output (kernel_filters x (input_features-kernel_features)/s+1)
- $x$ = input (input_filters x input_features)
- $l$ = kernel index
- $j$ (2D) or $(j1,j2)$ (3D) $=$ feature index
- $s$ (2D) or $(s1,s2)$ (3D) $=$ stride
- $f$ (2D) or $(f1,2)$ (3D) = kernel features

~~~python
AveragePooling_2D(
    input_features = 94,
    kernel_features = 2,
    stride = 2)
~~~

~~~python
AveragePooling_3D(
    input_features = (8,8),
    kernel_features = (2,2),
    stride = (2,2))
~~~

## Activation Functions

**Sigmoid**

$f(x) = \frac{1}{1+e^{-x}}$

~~~python
Sigmoid()
~~~

**Tanh**

$f(x) = \frac{2}{1+e^{-2x}} - 1$

~~~python
Tanh()
~~~

**ReLu**

$f(x) = max(0,x)$

~~~python
ReLu()
~~~

## Loss Functions

- $y$ = true value (n x 1)
- $\hat{y}$ = predicted value (n x 1)

**Mean Square Error**

$loss(y,\hat{y}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$

~~~python
MeanSquareError()
~~~

**Binary Cross Entropy With Optional Weights**

$loss(y,\hat{y}) = \frac{1}{n} \sum_{i=1}^{n} w_0(1-y_i)ln(1-\hat{y}_i) + w_1(y_i)ln(\hat{y}_i)$  

- $(w_0,w_1)$ = weights

~~~python
BinaryCrossEntropy(
    weights = (1,1))
~~~

**Categorical Cross Entropy**

$softmax(x) = \frac{exp(x-max(x))}{exp(\sum_{j=1}^{f} x_j-max(x))}$\
$loss(y,\hat{y}) = \frac{1}{n} \sum_{i=1}^{n} -ln(\hat{y}_{y_i})$
  - $x$ = input (1 x f)

~~~python
CategoricalCrossEntropyWithSoftmax()
~~~

## Optimizers

  - $p$ = parameter value
  - $lr$ = learning rate
  - $g$ = parameter gradient
  - $i$ = batch index

**Standard Gradient Descent (SGD)**

1. $p_{i+1} = p_{i} - lr*g$

~~~python
Stochastic_GD(
    learning_rate = .05,
    friction = 0,
    nesterov = False)
~~~

**Stochastic Gradient Descent with Momentum**

1. if nesterov: $p_i=p_i+\mu v_i$
2. $v_{i+1} = \mu v_i + lr*g$
3. $p_{i+1} = p_{i}-v_{i+1}$
    - $\mu$ = friction
    - $v$ = velocity
    - $\mu = 0 \: \rightarrow \:$ Standard Gradient Descent

~~~python
Stochastic_GD(
    learning_rate = .01,
    friction = 0.9,
    nesterov = False)
~~~

**AdaGrad**

1. $c_{i+1} = c_i+g^2$
2. $p_{i+1} = p_i - lr \frac{g}{\sqrt{c_{i+1}}}$
    - $c$ = cache

~~~python
AdaGrad(
    learning_rate = .01)
~~~

**RMSProp**

1. $c_{i+1} = d c_i+(1-d) g^2$
2. $p_{i+1} = p_i - lr \frac{g}{\sqrt{c_{i+1}}}$
    - $c$ = cache
    - $d$ = decay rate

~~~python
RMSProp(
    learning_rate = .01,
    decay_rate = 0.9)
~~~

**Adam**

1. $c_{i+1} = \beta_1 c_i + (1-\beta_1) g$
2. $mt = \frac{c_i}{1-\beta_1^i}$
3. $v_{i+1} = \beta_2 v_i + (1-\beta_2) g^2$
4. $vt = \frac{v_{i+1}}{1-\beta_2^i}$
5. $p_{i+1} = p_i - lr \frac{mt}{\sqrt{vt}}$
    - $c$ = cache
    - $v$ = velocity
    - $mt$ = adjusted momentum
    - $vt$ = adjusted velocity
    - $\beta_1$ = momentum parameter
    - $\beta_2$ = velocity parameter

~~~python
Adam(
    learning_rate = .01,
    beta1 = .9,
    beta2 = .999)
~~~

## Neural Network

High level wrapper for a neural network model.

~~~python
NeuralNetwork(
    name = 'Three Layer Network',
    layers = [
        Linear(
            input_features = 188,
            output_features = 3),
        Sigmoid(),
        Linear(
            input_features = 3,
            output_features = 1),
        Sigmoid()],
    loss_obj = BinaryCrossEntropy(weights=(1,23)),
    optimizer_obj = Stochastic_GD(learning_rate=.05))
~~~

~~~python
NeuralNetwork(
    name = 'Convolutional Neural Network',
    layers = [
        Conv_3D(
            input_filters = 1,
            input_features = (28,28),
            kernel_filters = 24,
            kernel_features = (5,5),
            stride = (1,1),
            padding = 0),
        Sigmoid(),
        MaxPooling_3D(
            input_features = (24,24) ,
            kernel_features = (2,2),
            stride = (2,2)),
        Conv_3D(
            input_filters = 24,
            input_features = (12,12),
            kernel_filters = 48,
            kernel_features = (5,5),
            stride = (1,1),
            padding = 0),
        Sigmoid(),
        MaxPooling_3D(
            input_features = (8,8),
            kernel_features = (2,2),
            stride = (2,2)),
        Conv_3D(
            input_filters = 48,
            input_features = (4,4),
            kernel_filters = 64,
            kernel_features = (3,3),
            stride = (1,1),
            padding = 0),
        Sigmoid(),
        Reshape(256),
        Linear(
            input_features = 256,
            output_features = 10)],
    loss_obj = CategoricalCrossEntropyWithSoftmax(),
    optimizer_obj = Stochastic_GD(learning_rate=.05))
~~~

~~~python
NeuralNetwork(
    name = 'Graph Convolution Network',
    layers = [
        GraphConv(adjacency_matrix=adj_mats[0],input_features=805,output_features=400),
        Sigmoid(),
        GraphConv(adjacency_matrix=adj_mats[1],input_features=400,output_features=100),
        Sigmoid(),
        GraphConv(adjacency_matrix=adj_mats[2],input_features=100,output_features=1),
        Sigmoid(),
        Reshape(9),
        Linear(input_features=9,output_features=1),
        Sigmoid()],
    loss_obj = BinaryCrossEntropy(weights=(1,1)),
    optimizer_obj = Stochastic_GD(learning_rate=.05,friction=.8,nesterov=False))
~~~