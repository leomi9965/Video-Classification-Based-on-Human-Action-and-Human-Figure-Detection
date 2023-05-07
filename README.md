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

Lastly, we use our feature extractor and classification network to perform training, and test on the test data. We then build a single video test function which takes any single video and convert it into feature and feed the feature to the network, it will return the corresponding percentage for each action category. The action with highest percentage will be our predicted label.

For the human figure detection part, the main steps include loading a pre-trained object detection model, extracting video frames, detecting humans in the frames, drawing bounding boxes around the detected humans, and creating a new video with the processed frames.

First, we set up the environment by installing the necessary Python libraries, including TensorFlow, OpenCV, and NumPy, and so on.

Second, we download the ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8 model as the pre-trained model from the TensorFlow Model Zoo and extract the files. We then train the model by using the provided load_model() function to load the pre-trained model into the project.

Third, we prepare the video by selecting the input video file that contains the dancing people for human frame detection. We then specify the input video path, output paths for the extracted frames, processed frames, and the final output video.

Fourth, we extract video metadata by using the get_video_duration(), and get_video_fps_and_size() functions to obtain the video's duration, FPS, and frame size. We then utilize the extract_frames() function to convert the input video into individual frames.

Finally, we process frames by detecting humans and drawing bounding boxes around them in the extracted frames by using the process_frames() function. Then we create the output video: merge the processed frames back into a video using the frames_to_video() function. The output video will have the same FPS and frame size as the original video.
