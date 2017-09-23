#**Behavioral Cloning** 

[//]: # (Image References)

[image1]: ./examples/center.png 
[image2]: ./examples/left.png 
[image3]: ./examples/right.png 
[image4]: ./examples/track2.png 
[image5]: ./examples/translated.png 
[image6]: ./examples/shadowed.png 



**Behavioral Cloning Project**


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

There are two tracks in simulator, which could be assessed as easy and more difficult. Easy track is mostly straight 
with only only a few bends and is flat, while the other track (optional) is very curvy and also contains U-turns. Besides,
it has different levels of height and only in few parts the surface is flat. Most time consuming part of this project was making the car drive on  the second track.
 
 There are three cameras attached to the car: left, center and right and we get RGB images of size (160, 320). We gather data by driving the car in manual mode. 
 Data contains steering angle (from interval [-1, 1] ), brake, throttle and speed values for each frame made by central camera. To get an idea of 
 how the roads look like, we can see the images below, which also contain the steering angles:
 
 ![image1]
 
 ![image2]
 
 ![image3]
 
 And the second track: 
 
 ![image4]

To gather the data, first, I have driven the car on both tracks, both clockwise and counter-clockwise. I also use the images
 from left and right cameras with a correction of 0.2 and -0.2 respectively. Then I add random horizontal and vertical translations
 to the images. Vertically, images are shifted by pixels between -20 and 20. Horizontally, they are shifted by pixels from
   interval of [-40, 40]. Since the steering angle should be corrected for the horizontal shifts I use the formula
   
```
    angle + correction * horizontal_shift / max_horizontal_shift
```

where
```
horizontal_shift = max_horizontal_shift * uniform(-1, 1)
```

and
```
correction = numpy.abs(horizontal_shift / 100)
```

Here are some examples of the images after the random shifts:

![image5]

Second track contains several regions with shadow. To make the model learn how to behave in shady regions, we need more 
of these kind of samples. To artificially increase the portion of shady images, I convert randomly picked images to HLS (hue, lightness, saturation) 
color space, select a random area in the image and reduce value of lightness. The coefficient for reducing lightness is 
also sampled uniformly from interval [0.4. 0.6]. 

![image6]

To increase the data and to remove the bias towards certain side for the tracks I also flip the images and of course, 
multiply the corresponding steering angle by -1. 

###Model Architecture and Training Strategy

|Layer (type) |Output Shape|Param #|
--------------|:-----------:|------ |
|input_1 (InputLayer)| (None, 160, 320, 3)|0|
lambda_1 (Lambda)            |(None, 160, 320, 3)       |0|         
conv2d_1 (Conv2D)            |(None, 160, 320, 32)     | 2432  |    
max_pooling2d_1 (MaxPooling2) |(None, 80, 160, 32)      | 0     |    
conv2d_2 (Conv2D)            |(None, 80, 160, 48)      | 38448 |    
max_pooling2d_2 (MaxPooling2)|(None, 40, 80, 48)       | 0     |    
conv2d_3 (Conv2D)            |(None, 36, 76, 64)       | 76864 |    
max_pooling2d_3 (MaxPooling2) |(None, 18, 38, 64)       | 0     |    
conv2d_4 (Conv2D)            |(None, 14, 34, 64)       | 102464|    
max_pooling2d_4 (MaxPooling2) |(None, 7, 17, 64)        | 0     |    
conv2d_5 (Conv2D)            |(None, 3, 13, 64)        | 102464|    
max_pooling2d_5 (MaxPooling2) |(None, 1, 6, 64)         | 0     |    
flatten_1 (Flatten)          |(None, 384)              | 0     |    
dense_1 (Dense)              |(None, 100)              | 38500 |    
dense_2 (Dense)              |(None, 100)              | 10100 |    
dropout_1 (Dropout)          |(None, 100)              | 0     |    
dense_3 (Dense)              |(None, 1)                | 101   |    

 371,373 parameters

[![asdasd](http://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](http://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID_HERE)


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
