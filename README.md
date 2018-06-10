# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model WITHOUT using generator
* model_gen.py containing the script to create and train the model USING generator
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py and model_gen file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is using the NVIDIA model introduced in the course. It consists of 5 CNN layers and 3 full connection layers (model.py lines 43-55; model_gen.py lines 60-72) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains one dropout layer in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py code line 58; model_gen.py line 52). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 57, model_gen.py line 74).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. As I am a very bad racing car game player, I just use the training data provided by Udacity.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I use the NVIDIA model as the start point of my solution. The model works well. The only change I made is adding one Dropout Layer with 0.5 keep rate.

My Video card does not come across memory issue in this project, however I choose both approaches (using / not using generator) for a better understanding of the generator. I think the code sample in the course should be changed a bit:

```python
model.fit_generator(train_generator, validation_data=validate_generator, steps_per_epoch=len(train_lines)/batch_size, epochs=5, validation_steps=len(validate_lines)/batch_size)
```
The steps_per_epoch should be the total number divided by batch_size.  

#### 2. Final Model Architecture

The final model architecture (model.py lines 43-55; model_gen.py lines 60-72) consisted of 5 CNN Layers with 3 Full Connection Layers and 1 Dropout Layer.

#### 3. Creation of the Training Set & Training Process

At first, I made a mistake by treating 3 cameras the same way (no correction being used). The result was awful - the trained model didn't steer the car at all. Then I reviewed the course and was aware of the mistake. After applied the 0.2 correction the model worked as expected.

Also I used augmented data to increase the data (using the approach suggested in the course which is to mirror the images)

After the collection process, I had 48216 training data points. I then normalise the data by using Lamdba layer.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3-5 as evidenced by the validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
