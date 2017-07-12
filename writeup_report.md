# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center]: ./examples/center.jpg
[left_side]: ./examples/left_side.jpg
[right_side]: ./examples/right_side.jpg
[before_flip]: ./examples/before_flip.jpg
[after_flip]: ./examples/after_flip.jpg
[bridge1]: ./examples/bridge1.jpg
[bridge2]: ./examples/bridge2.jpg

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is mostly following network architecture in Nvidia's paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). Simply speaking, it consists of 5 convolution layers with 5x5 or 3x3 filter sizes and depths between 24 and 64, and followed by 3 full connection layers. (model.py lines 61-74). The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 54).

The only difference between my model and the Nvidia paper model is that, I added extra dropout layers after conv layers to reduce overfitting. Please see the layer details on the next section.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers after each convolution layer in order to reduce overfitting. And the model also uses L2 regulation, which turns out to be very effective.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 78). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 77).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, especially on some uncommon road condition like bridge and sharp turns.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the Nvdia' paper *End-to-End Deep Learning for Self-Driving Cars*. I thought this model might be appropriate because the problem domain is similar, though the case for this project might be simplier.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model was getting lower and lower mean squared error after each epoch, while the error on validation set was high and didn't change much after epochs. This implied that the model was overfitting. 

To reduce the overfitting, I added dropout layers after each conv layers, which turned out to be effective. (I also tried dropout after full connection layers, but that didn't help.)

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, like at the begining and ending part of the bridge, and at a sharp right turn near the end of the lap. I think that's mostly because of lacking in data. For example, if you simply drive a lap, there are only a few images with paved road at the bottom and bridge at the top, not enough to train a reliable model in such scenario. And because most of the turns are left-turns, so the model didn't do well for the right-turn case.

To improve the driving behavior in these cases, I intentionally recorded more data at both ends of the bridge, and more data on right turns. And of course, we also need the recovery data, i.e., driving from the left and right borders of the road to the center. I found that collecting more data for particular bad scenarios usually gave me better results than tuning the model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		      |     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		    | 160x320x3 RGB image   							| 
| Normalization         | simply (x / 255.0 - 0.5)            |
| Cropping              | Crop top 50 pixel and bottom 20 pixel |
| Convolution 5x5     	| 2x2 stride, valid padding, 24 output channels, RELU activation 	|
| Dropout               | drop 20%              |
| Convolution 5x5	      | 2x2 stride, valid padding, 36 output channels, RELU activation    |
| Dropout               | drop 20%              |
| Convolution 5x5     	| 2x2 stride, valid padding, 48 output channels, RELU activation 	|
| Dropout               | drop 20%              |
| Convolution 3x3     	| 2x2 stride, valid padding, 64 output channels, RELU activation 	|
| Dropout               | drop 20%              |
| Convolution 3x3     	| 2x2 stride, valid padding, 64 output channels, RELU activation 	|
| Flatten               |        | 
| Fully connected		    | Dense(100), RELU activation      |
| Fully connected		    | Dense(50), RELU activation       |
| Fully connected		    | Dense(10), RELU activation       |
| Output                     |                        |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track one using center lane driving - two of them are clockwise, while the other two laps are counter-clockwise. Here is an example image of center lane driving:

![alt text][center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center, like the following examples:

![alt text][left_side]
![alt text][right_side]

To augment the data set, I also flipped images and angles thinking that this would reduce overfit. For example, here is an image that has then been flipped:

![alt text][before_flip]
![alt text][after_flip]

Like described above, I particularly generated more data on the bad cases during the testing iterations, for example, on the bridge:

![alt text][bridge1]
![alt text][bridge2]

After the collection process, I had about 10K of data points. I noticed that a large amount of data has steering angle 0, so I downsampled the data points whose angle is very small (<0.1) by half. I believe that will make the data more balanced. After downsampling and flipping, I got about 13K data points finally.

I didn't use the image from the left and right camera, because I found it's not easy to get a good correction value to adjust the steering angle - either nothing get changed, or the car will wiggle on the road.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 to 10 based on the output of `model.fit()`. I used an adam optimizer so that manually training the learning rate wasn't necessary.
