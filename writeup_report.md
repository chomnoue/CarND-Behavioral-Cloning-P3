#**Behavioral Cloning** 

##Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nvidia_architecture]: ./model.png "Model Architecture"
[center_lane_driving]: ./examples/center_lane_driving.jpg "Center Lane driving"
[recovering_from_left_side]: ./examples/recovering_from_left_side.jpg "Recovery from left side"
[recovering_from_right_side]: ./examples/recovering_from_right_side.jpg "Recovery right side"
[center_lane_driving_flipped]: ./examples/center_lane_driving_flipped.jpg "Center land driving flipped"


---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 showing how the car is self-driving on a lap of the circuit

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. 
It takes as parameters the name of the architecture, the number of epochs to run and the correction value to be applied to the angle for the left and right cameras images

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I have tried two model architectures: 

* one from [comma.ai](https://github.com/commaai/research/blob/master/train_steering_model.py). It consists of 3 convolutional layers (16*8*8, 32*5*5 and 64*5*5) folowed  by 2 fully conncted layers (512 and 1). See get_model() in model.py
* the other one is from [nvidia](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). It consists of 6 convolutional layers (24*5*5, 36*5*5, 48*5*5, 64*3*3 and 64*3*3), followed by 4 fully connected layers (100, 50, 10, 1). See get_nvdia_model()

Both models include RELU layers to introduce nonlinearity.

The images are cropped to remove the parts zones non-relevant to the driving decision (lines 57 and 77) and then normalized using a Keras lambda layer (lines 58 and 78). 

The nvidia architecture produced better results so I used it for the released model.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 65, 68, 86, 88 and 90). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 106-120). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 177).

####4. Appropriate training data

I used a combination of center lane driving, recovering from the left and right sides of the road (model.py lines 35-44). 
I have tried values .1, .2, .3 ,.4 and .5 values for the *correction*. Value .4 better results in the simulator so I kept it for the final model 
I also added, for each image, a flipped version of it (code line 28-29)


###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to look for existing working architectures and try to adapt one of them

So I chose the nvidia architecture as explained above. I thought this model might be appropriate because it has already been used to train self driving cars.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

To combat the overfitting, I modified the model by adding droput layers.

Then I augmented training data by using the left and right cameras, and adding a flipped version of each image.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I played with the correction applied to the angle value for left and right cameras images. The value of .4 showed to be the best.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: The parameters of each layers are discribed above)

![Mocel architecture][nvidia_architecture]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center land driving][center_lane_driving]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to avoid crossing the lanes These images show what a recovery looks like :

![Recovering from left side][recovering_from_left_side]
![Recovering from right side][recovering_from_right_side]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would help the model to generalize. For example, here the above center lane driving image that has then been flipped:

![Center lane driving flipped][center_lane_driving_flipped]

I also used for each image from right and left cameras to he help the car learn to stay away from lanes.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The result was quite good after 20 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
