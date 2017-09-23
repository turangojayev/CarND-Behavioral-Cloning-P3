[image1]: ./examples/center.png 
[image2]: ./examples/left.png 
[image3]: ./examples/right.png 
[image4]: ./examples/track2.png 
[image5]: ./examples/translated.png 
[image6]: ./examples/shadowed.png 



**Behavioral Cloning Project**
----------------------------------------

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
 from left and right cameras with a correction of 0.2 and -0.2 respectively. I tried to make more of the images at sharp turns,
  thus have driven the car to certain difficult parts on second track and recorded starting from those points, while keeping the speed quite low.Then I add random horizontal and vertical translations
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
of these kind of samples. To artificially increase the portion of shady images, I convert randomly (uniformly, with probability of 0.5)
picked images to HLS (hue, lightness, saturation) color space, select a random area in the image and reduce the value of lightness. The coefficient for reducing lightness is 
also sampled uniformly from interval [0.4. 0.6]. For example, shadow added to random areas of the images above would look like this:

![image6]

To increase the data further and to remove the bias towards certain side for the tracks I also flip the images horizontally and of course, 
multiply the corresponding steering angle by -1. After all this procedures I would have ended up with 206418 images. I keep 
only the paths to the images and use python generators to read the images only when they are needed. 
Random shifts-shadows are done on the fly. Thus, the actual number of distinct images is more than 206418 
(translation and shadowing are done randomly every time the image is read).

###Model Architecture and Training Strategy

Inspired by [NVIDIA's end to end approach](https://arxiv.org/pdf/1604.07316.pdf) I used similar architecture to solve this problem.
First I normalize the images by subtracting 127 and dividing by 128, however, I neither cropped any part of the image, nor
resized them. The difference from the NVIDIA's architecture is in the stride sizes (I have 1x1 strides), usage of max pooling
after convolutional layers, number of filters at each convolutional layer, kernel sizes of the last two convolutional layers 
(5x5 instead of 3x3) and in the number and sizes of the fully connected layers (2 instead of 3 layers, each of size 100).
 In total, my model had 371,373 parameters. The number of parameters are listed in the table below:

|Layer (type) |Output Shape|Param #|
--------------|:-----------:|------ |
|input_1 (InputLayer)| (None, 160, 320, 3)|0|
lambda_1 (Lambda)            |(None, 160, 320, 3)       |0|         
conv2d_1 (Conv2D), kernel_size=5x5 |(None, 160, 320, 32)     | 2432  |    
max_pooling2d_1 (MaxPooling2) |(None, 80, 160, 32)      | 0     |    
conv2d_2 (Conv2D) ,kernel_size=5x5|(None, 80, 160, 48)      | 38448 |    
max_pooling2d_2 (MaxPooling2)|(None, 40, 80, 48)       | 0     |    
conv2d_3 (Conv2D), kernel_size=5x5|(None, 36, 76, 64)       | 76864 |    
max_pooling2d_3 (MaxPooling2) |(None, 18, 38, 64)       | 0     |    
conv2d_4 (Conv2D), kernel_size=5x5|(None, 14, 34, 64)       | 102464|    
max_pooling2d_4 (MaxPooling2) |(None, 7, 17, 64)        | 0     |    
conv2d_5 (Conv2D), kernel_size=5x5|(None, 3, 13, 64)        | 102464|    
max_pooling2d_5 (MaxPooling2) |(None, 1, 6, 64)         | 0     |    
flatten_1 (Flatten)          |(None, 384)              | 0     |    
dense_1 (Dense)              |(None, 100)              | 38500 |    
dense_2 (Dense)              |(None, 100)              | 10100 |    
dropout_1 (Dropout)          |(None, 100)              | 0     |    
dense_3 (Dense)              |(None, 1)                | 101   |    

 I use rectified linear unit as activation function and it performed better than 
 exponential linear unit in my experiments. I trained my model for 4 epochs with adaptive momentum (adam) optimizer and batch size of 32. Initial learning rate of 0.001 ended up with making huge jumps
for the loss and setting it to 0.0001 has given me the best results.

To track the value of loss function during training, I split the data into train (80%) and validation (20%) parts.
To make sure that both train and validation data contain images with corresponding steering angle from the whole interval, 
I use stratified sampling. For that, I group the steering angle values by putting them into separate bins, given by values
[-1, -0.8, -0.6, -0.3, 0.3, 0.6, 0.8, 1]. After all these steps my model learnt how to drive the car on both tracks at 
speed of 30 miles per hour.


[Link to autonomous driving on track1](https://www.youtube.com/watch?v=FyK2CDwMjvI)

[Link to autonomous driving on track2](https://www.youtube.com/watch?v=d4q78V76Xlo)