# How facenet/other facial recognition pipelines work
- - - -
## What is FaceNet?
FaceNet is currently a state-of-art face recognition, verification and clustering neural network. It’s depth is of 22 layers that trains its output to be a 128-dimensional embedding. It also uses a loss function called **triplet loss**.
![](https://miro.medium.com/max/1400/1*ZD-mw2aUQfFwCLS3cV2rGA.png)
Image from [FaceNet](https://arxiv.org/pdf/1503.03832.pdf?source=post_page---------------------------)

As you can see above, the network consists of a batch input layer and a deep CNN followed by L2 normalisation, which results in a face embedding. It is then followed by triplet loss during training.

It is important to mention that, quoting directly from the source:

> **Triplet Loss** minimises the distance between an *anchor* and a *positive*, both of which have the same identity, and maximises the distance between the *anchor* and a *negative* of a different identity.  

![](FaceNet/Screenshot%202019-07-26%20at%2014.31.44.png)

The FaceNet model currently has two CNN architectures, the Zeiler and Fergus model:
![](FaceNet/Screenshot%202019-07-26%20at%2014.28.35.png)
Image taken from Zeiler & Fergus

And the GoogleNet’s:
￼![](https://mohitjainweb.files.wordpress.com/2018/06/googlenet-architecture-table.png)

The most used one being GoogleNet, which has the codename Inception.
- - - -
## FaceNet Performance
![](FaceNet/Screenshot%202019-07-26%20at%2014.33.56.png)
This table compares the performance of the different architectures. It also shows the standard error of the mean.
- - - -
## Inception Module
One of the main features of GoogleNet is the **Inception Module**.

The module solves the problem of deciding what type of convolution to do at each layer. When instantiating convolutional layers, one has to decide: will it be a 3x3 convolution? 5x5? Before the Inception module, there was no method of finding out what is the best layer combination for the network.

GoogleNet proposes that instead of having to decide the type of convolution, you can use all of them and let the network decide by itself the optimal configuration. The way this happens is by doing each convolution in parallel on the same input, with same padding and concatenating the feature maps from each convolution into a feature map. Finally, the feature map is fed as input to the next Inception module.

The Inception architecture is restricted to filter sizes of 1x1, 3x3 and 5x5. The smalls filters help capture local details and features. Additionally, a 3x3 max pooling was also added as it has been found that pooling layers make the network work better.
![](https://mohitjainweb.files.wordpress.com/2018/06/inception-module-naive1.png)

One constraint of this module is that the large convolutions end up being computationally expensive. For example, if there is an input of 28x28x192 and it goes through a 5x5 convolution with 32 filters, the number of operations would be of 120,422,400 (5x5x192x32x28x28), meaning a lot of computation. However, this problem was solved with **dimensionality reduction**. Dimensionality reduction involves adding 1x1 convolutions before large convolutions. Therefore, applying dimensionality reduction to the previous problem: the input of 28x28x192 goes through a 1x1 convolution with 16 filters would result in a 2,408,448 operations and an output of 28x28x16. The output then is fed through the actual convolution, 5x5 with 32 filters, meaning 10,035,200 operations (5x5x16x32x28x28), giving an output of 28x28x32. Adding the number of operations together we get 12,443,648, a reduction by about a factor of 10 comparing to not applying dimensionality reduction.
![](https://mohitjainweb.files.wordpress.com/2018/06/5x5-convolution-with-dimensionality-reduction-e1528493108712.png)

The model would now looks like this:
![](https://mohitjainweb.files.wordpress.com/2018/06/inception-module-with-dimensionality-reduction.png?w=1398)
- - - -
## Deep Network
Some of the features of that makes GoogleNet a ImageNet 2014 challenge winner:

* 22 layers compared to 8 layers AlexNet.
* Computational cost of 2 times less than AlexNet.
* More accurate than AlexNet.
* Low memory usage and power consumption.
* Smaller number of parameters, 12 times less than AlexNet.

Now analysing the architecture, as said before, the network is 22 layers deep. The initial layers are simple convolution layers. However, after that there are multiple blocks of *inception modules*
