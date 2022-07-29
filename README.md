# Computer Vision: Vehicle Type Identification

Using Torch to perform our machine learning model, Deep convolutional neural network (DCNN) is used to recognize whether a car is an EV/non-EV when given an image. 

## Running the code
To test the finished code with the checkpoint, kindly run EV_Detect_test.ipynb whereas to see the full code for the whole model, open Challenge_2.ipynb for details. 
From the code, these two images can be detected whether is an EV/non EV car.

BYD Car detected as EV car

![BYD_Seal (EV)](https://user-images.githubusercontent.com/93107581/179144066-4f628ae3-a3cb-460e-b181-f53c7914754e.jpg)

Myvi detected as non EV car

![Myvi (Non EV)](https://user-images.githubusercontent.com/93107581/179144179-b1c52963-b9eb-4e27-960d-4801c393cf05.png)


## Why DCNN?

One of the weaknesses of an ordinary feedforward neural network with fully connected layers is that it has no prior inbuilt assumption about the data it is supposed to learn from. Hence, comes deep learning. Deep learning is a machine learning technique used to build artificial intelligence (AI) systems, based on the concept of artificial neural networks (ANN), which use several layers of neurons to process enormous quantities of data to do complicated analysis. The most popular form of deep convolutional neural network (CNN or DCNN) for pattern recognition in pictures and videos.Developed from conventional artificial neural networks, DCNNs use a three-dimensional neural pattern inspired by the visual brain of animals. 

The strength of DCNNs is in their layering. A DCNN uses a three-dimensional neural network to process the Red, Green, and Blue elements of the image at the same time. This considerably reduces the number of artificial neurons required to process an image, compared to traditional feed forward neural networks. Deep convolutional neural networks receive images as an input and use them to train a classifier. The network employs a special mathematical operation called a “convolution” instead of matrix multiplication.

The architecture of a convolutional network typically consists of four types of layers: convolution, pooling, activation, and fully connected.

![image](https://user-images.githubusercontent.com/93107581/179124491-18c2074d-cd3b-4ab4-bef4-53b99c3c3ecb.png)

### DCNN Architecture
For this particular case, we are utilizing Densenet architecture. The architecture explains how each layer receives data from prior layers' output feature maps. Because each layer receives more supervision from the preceding layer as a result of this severe residual reuse, the loss function will respond appropriately, making the network more powerful.

In pyTorch, pre-trained models are available and these models allow others to quickly obtain cutting-edge results in computer vision without needing such large amounts of computer power, patience, and time. 

DenseNet consists of 2 blocks:

1. Dense Block
A Dense Block is a module used in convolutional neural networks that connects all layers (with matching feature-map sizes) directly with each other. It was originally proposed as part of the DenseNet architecture. The layers are Batch Normalization, ReLU Activation and 3x3 Convolution.

2. Transition Layer.
In ResNet sum of residual will be performed, instead of summing residual Densenet concatenates all the feature maps. This layer is made of Batch Normalization, 1x1 Convolution and Average Pooling.


## Reference and extra reading:
1. https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803
2. https://www.run.ai/guides/deep-learning-for-computer-vision/deep-convolutional-neural-networks#:~:text=Deep%20convolutional%20neural%20networks%20(CNN,the%20visual%20cortex%20of%20animals.
3. http://www.diva-portal.org/smash/get/diva2:1111144/FULLTEXT02.pdf
4. https://towardsdatascience.com/architecture-comparison-of-alexnet-vggnet-resnet-inception-densenet-beb8b116866d
