# Video Classification Based On Human Action And Human Figure Detection:

# Installation:
numpy
pandas
matplotlib
os
tensorflow
imutils
imageio
cv2

# Usage:
In order to use this document, you will need to have your own video datasets and split them into two groups: train and test.
The potential website for you to collect data is: https://www.pexels.com/zh-cn/ 


# General guidelines: 
First, we have function that crop image into square shape to remove irrelevant part for our analysis. Then, we have another function that load the video into the network. For this function, we used the function VideoCapture() from Open CV, and it will read the frames of the video and then we applied the first function to make sure each frame is in the correct size. 

Second, we build our feature extractor that extract features from each frame. We used InceptionV3 model from the package Keras. We applied the transfer learning and this model is a pre-trained model which trained on "Imagenet".

Third, we build a function that prepared all videos into features. First, we convert those videos into frames and then we shape it into correct sizes using the part 1's functions. However, every video has different length, we have to make sure the input features to the network all have the same length. Therefore, we applied padding techniques. We first choose a max sequence length, and if a video's length is smaller than the max sequence length, we set 0 to those parts. In this way, we can make sure all video has the equal number of frames for the network to analyze.

Fourth, we build our sequence models that take the features as input and return probabilities for each action category.For the sequence model, it is a RNN architecture with two GRU layers, a dropout layer and a dense layer, For the last layer, we choose the activation function as softmax. In the end, it will return the corresponding probabilities for each action category. The detailed pipeline for feature extraction and classification is listed below:

Lastly, we using our feature extractor and classification network to perform training, and testing on the test data. We then build a single video test function which takes any single video and convert it into feature and feed the feature to the network, it will return the corresponding percentage for each action category. The action with highest percentage will be our predicted label.
