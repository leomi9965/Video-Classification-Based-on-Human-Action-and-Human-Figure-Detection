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
First, we have a function that crops the image into a square shape to remove the irrelevant part for our analysis. Then, we have another function that loads the video into the network. For this function, we used the function VideoCapture() from Open CV, which will read the video frames; then, we applied the first function to make sure each frame is in the correct size. 

Second, we build our feature extractor that extracts features from each frame. We used the InceptionV3 model from the package Keras. We applied transfer learning, and this model is a pre-trained model trained on "Imagenet."

Third, we built a function that prepared all videos into features. First, we convert those videos into frames and then shape them into the correct sizes using part 1's functions. However, every video has a different length; we have to ensure the input features to the network all have the same length. Therefore, we applied padding techniques. We first choose a max sequence length, and if a video's length is smaller than the max sequence length, we set 0 to those parts. In this way, we can ensure all video has an equal number of frames for the network to analyze.

Fourth, we build our sequence models that take the features as input and return probabilities for each action category. The sequence model is an RNN architecture with two GRU layers, a dropout layer, and a dense layer; For the last layer, we choose the activation function as softmax. Ultimately, it will return the corresponding probabilities for each action category. The detailed pipeline for feature extraction and classification are listed below:

Lastly, we use our feature extractor and classification network to perform training and test on the test data. We then build a single video test function that converts any single video into a feature and feeds the feature to the network; it will return the corresponding percentage for each action category. The action with the highest percentage will be our predicted label.

For the human figure detection part, the main steps include loading a pre-trained object detection model, extracting video frames, detecting humans in the frames, drawing bounding boxes around the detected humans, and creating a new video with the processed frames.

First, we set up the environment by installing the necessary Python libraries, including TensorFlow, OpenCV, NumPy, etc.

Second, we download the ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8 model as the pre-trained model from the TensorFlow Model Zoo and extract the files. We then train the model using the provided load_model() function to load the pre-trained model into the project.

Third, we prepare the video by selecting the input file containing the dancing people for human frame detection. We then specify the input video path and output paths for the extracted, processed, and final output video.

Fourth, we extract video metadata using the get_video_duration() and get_video_fps_and_size() functions to obtain the video's duration, FPS, and frame size. We then utilize the extract_frames() function to convert the input video into individual frames.

Finally, we process frames by detecting humans and drawing bounding boxes around them in the extracted frames by using the process_frames() function. Then we create the output video: merge the processed frames into a video using the frames_to_video() function. The output video will have the same FPS and frame size as the original video.
