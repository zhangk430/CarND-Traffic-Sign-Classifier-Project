# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/hist.png "Visualization"
[image4]: ./examples/00036.png "Traffic Sign 1"
[image5]: ./examples/00037.png "Traffic Sign 2"
[image6]: ./examples/00038.png "Traffic Sign 3"
[image7]: ./examples/00039.png "Traffic Sign 4"
[image8]: ./examples/00112.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/zhangk430/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the number of images per each category. The most frequent image with its category is also shown. 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)
I decided to normalize the images by subtracting each pixel by 128 and dividing by 128 since the range of each pixel is [0,255] so by normalizing it's range become [-1,1]. After normalization, the training converges much faster. I also tried to convert the images to grayscale but the result seems worse. I think the color information is valuable so converting to grayscale image seems not a good idea.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU+dropout			|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU+dropout          |                                               |
| Max pooling           | 2x2 stride, valid padding, outputs 5x5x16     |
| Fully connected		| 400 -> 120        						    |
| RELU+dropout          |                                               |
| Fully connected       | 120 -> 84                                     |
| RELU+dropout          |                                               |
| Fully connected       | 84 -> 43                                      |
| RELU+drop             |                                               |
| Softmax				|            									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the adam optimizer to optimize the cross entropy. The batch size I use is 128 and the number of epochs 50. I choose 0.001 as the learning rate. The probability for dropout is 0.5.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.988
* validation set accuracy of 0.940
* test set accuracy of 0.938

* What architecture was chosen?
I chose the LeNet-5 as the architecture.
* Why did you believe it would be relevant to the traffic sign application?
LeNet performs very well for character recognition. Traffic sign application is very similar to the character recognition since it's also a multi-class classification problem. Given the traffic sign images, it classifies it as 43 different classes. So I think LeNet will also perform well for traffic sign.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The training, validation and test accuracy are 0.987, 0.941 and 0.933. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The second and fourth images might be difficult to classify because they are very dark and hard to recognize.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority Road      	| Priority Road   								| 
| Beware of ice/snow    | Beware of ice/snow							|
| 70 km/h				| 70 km/h										|
| General Caution	    | General Caution					 			|
| No vehicles			| No vehicles      						    	|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.933.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a beware of ice/snow (probability of 0.35), and the image does contain a beware of ice/snow sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .35         			| Beware of ice/snow   							| 
| .17     				| Dangerous curve to the right 					|
| .12					| Right-of-way at the next intersection			|
| .07	      			| Slippery road					 				|
| .07				    | Children crossing      						|


For the second image, the model is very sure that this is a priority road (probability of 1), and the image does contain a priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Priority road   							    | 
| 0.0     				| Road narrows on the right 					|
| 0.0					| End of all speed and passing limits			|
| 0.0	      			| No passing for vehicles over 3.5 metric tons  |
| 0.0				    | Speed limit (70km/h)      					|

For the third image, the model is very sure that this is a Speed limit (70km/h) (probability of 1), and the image does contain a Speed limit (70km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (70km/h)   					    | 
| 0.0     				| Speed limit (30km/h) 					        |
| 0.0					| Speed limit (20km/h)			                |
| 0.0	      			| Speed limit (50km/h)                          |
| 0.0				    | Speed limit (80km/h)      					|


For the fourth image, the model is very sure that this is a General caution (probability of 0.96), and the image does contain a General caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.96         			| General caution   							| 
| 0.03     				| Traffic signals 					            |
| 0.0					| Pedestrians			                        |
| 0.0	      			| Road narrows on the right                     |
| 0.0				    | Road work      					            |

For the fifth image, the model is very sure that this is a priority road (probability of 1), and the image does contain a priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| No vehicles   							    | 
| 0.0     				| No passing 			                		|
| 0.0					| Yield			                                |
| 0.0	      			| Speed limit (50km/h)                          |
| 0.0				    | Speed limit (70km/h)      					|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


