# Quantisation
- - - -
## What is quantisation? 
Firstly it is important to understand that Neural Networks are composed of multiple layers of parameters. Each layer transforms the input image, separating and contracting features, resulting in the classification of input images to their respective classes. 
Deep learning for classification involves training parameters such that the algorithm learns to discern between different classes. This can be achieved by feeding a large number of labelled data to the network while updating the parameters to increase performance. However, a drawback is that a large number of parameters are used compared to more traditional approaches.
This is where quantisation comes into picture. It is a method to bring the neural network to a reasonable size while also achieving high performance accuracy. Quantisation can be defined as the process of approximating floating-point numbers to low bit width numbers, resulting in reductions of birth memory requirement and computational cost of using neural networks. 
- - - -
## Why is it necessary?
Why the need of quantisation? To be able to implement on-device applications, where we have memory and computation constraints. 
Additionally, it has been proven recently that training a single AI model can emit as much carbon as five cars in their lifetimes [4](https://www.technologyreview.com/s/613630/training-a-single-ai-model-can-emit-as-much-carbon-as-five-cars-in-their-lifetimes/?utm_medium=tr_social&utm_source=facebook&utm_campaign=site_visitor.unpaid.engagement&fbclid=IwAR04gQbmXY7OxR3vx8BRb52-pYhlqhXhxorXGSkgGkygu8jUVgbF_CJxYwI). Below you can see the estimated costs of training a model:
![](Quantisation/Screenshot%202019-07-26%20at%2011.44.50.png)
Image taken from [Technology Review](https://www.technologyreview.com/s/613630/training-a-single-ai-model-can-emit-as-much-carbon-as-five-cars-in-their-lifetimes/?utm_medium=tr_social&utm_source=facebook&utm_campaign=site_visitor.unpaid.engagement&fbclid=IwAR04gQbmXY7OxR3vx8BRb52-pYhlqhXhxorXGSkgGkygu8jUVgbF_CJxYwI) 

You can see below that using 16-bit floats vs 32-bit floats increases the energy efficiency by almost thrice, reinforcing the importance of quantising:
![](Quantisation/1*2clgI6r56PPLT3PV_0-Ssg.png)
Image taken from [Efficient Methods and Hardware for Deep Learning - Stanford University](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture15.pdf?source=post_page---------------------------)
- - - -
## Quantisation-aware training
It is possible to quantise a network after training is finished. However, the most effective method for retaining high accuracy  is to quantise **during** training.
The broad idea of quantisation-aware training is to cluster the weight values using K-mean clustering where the number of clusters is denied based on the number of bits desired. A codebook is created for each weight value, then used to perform a forward pass and get the gradients and finally update the centroid values of each cluster using the gradient updates.
![](Quantisation/1*VC1wz-68ZeQeHzor_3mhew.png)
Image taken from [Efficient Methods and Hardware for Deep Learning - Stanford University](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture15.pdf?source=post_page---------------------------)

After that, a forward pass is made using the centroid-replaced weight values, and the gradient are calculated accordingly. Since we know which cell indices belong to the same cluster, the gradient values get grouped based on the cluster the cell indices belong to, and then sum over the gradients to receive the gradient value for the cluster. Lastly, gradient descent is performed on the older centroid values using the gradient value obtained.
![](Quantisation/1*cPjdJFPB-sRmCejemWVE4w.png)
Image taken from [Efficient Methods and Hardware for Deep Learning - Stanford University](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture15.pdf?source=post_page---------------------------)

Now only the cluster indices and centroid values need to be stored. Suppose that 16 clusters are chosen, the weight matrix can be represented using only 4-bits, which is an 8x reduction in model size.

When referring to **latency**, quantised models are up to 2-4x faster on CPU and 4x smaller [3](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-post-training-integer-quantization-b4964a1ea9ba). Further speed-ups are expected with hardware accelerators, such as Edge TPUs.
![](Quantisation/0*P1uinlbEkK0lbJtq.png)
Image taken from [Model Optimization Tool - Tensorflow](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-post-training-integer-quantization-b4964a1ea9ba)

About **accuracy**, with just 100 calibration images from ImageNet dataset, fully quantised models have comparable accuracy with their float version:
![](Quantisation/1*jKJdkOme2Z4lFkcG0UEUQg.png)
Image taken from [Model Optimization Tool - Tensorflow](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-post-training-integer-quantization-b4964a1ea9ba)
- - - -
## Further Reading
### Quantisation
* Han, Song, Huizi Mão, and William J. Dally. [“Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding.”](https://arxiv.org/abs/1510.00149) arXiv preprint arXiv:1510.00149 (2015).
* Stanford University School of Engineering Lecture 15 on [Efficient Methods and Hardware for Deep Learning.](https://www.youtube.com/watch?v=eZdOkDtYMoo&source=post_page---------------------------)

### Quantised training
* Li H, De S, Xu Z, Studer C, Samet H, Goldstein T.  [“Training Quantized Nets: A Deeper Understanding.”](https://arxiv.org/abs/1706.02379)  Neural Information Processing Systems (NIPS), 2017 
* Gupta, Suyog, et al.  [“Deep learning with limited numerical precision.”](https://arxiv.org/abs/1502.02551)  International Conference on Machine Learning. 2015. 
* Courbariaux, Matthieu, Yoshua Bengio, and Jean-Pierre David.  [“Training deep neural networks with low precision multiplications.”](https://arxiv.org/abs/1412.7024)  arXiv preprint arXiv:1412.7024 (2014). 
* Wu, Shuang, et al.  [“Training and inference with integers in deep neural networks.”](https://arxiv.org/abs/1802.04680)  arXiv preprint arXiv:1802.04680 (2018).

### Quantisation in TF-Lite
* Manas Sahni blog post on 8-Bit Quantisation and TensorFlow-Lite: [Making Neural Nets Work With Low Precision | efficieNN sahnimanas.github.io](https://sahnimanas.github.io/2018/06/24/quantization-in-tf-lite.html?source=post_page---------------------------)
* Pete Warden’s blog posts on quantization: [1](https://petewarden.com/2015/05/23/why-are-eight-bits-enough-for-deep-neural-networks/?source=post_page---------------------------), [2](https://petewarden.com/2016/05/03/how-to-quantize-neural-networks-with-tensorflow/?source=post_page---------------------------), [3](https://petewarden.com/2017/06/22/what-ive-learned-about-neural-network-quantization/?source=post_page---------------------------) 
* Jacob B, Kligys S, Chen B, Zhu M, Tang M, Howard A, Adam H, Kalenichenko D.  [“Quantization and training of neural networks for efficient integer-arithmetic-only inference.”](https://arxiv.org/abs/1712.05877)  arXiv preprint arXiv:1712.05877. 2017 Dec 15.
* Tensorflow [Model Optimization Kit](https://www.tensorflow.org/model_optimization).

