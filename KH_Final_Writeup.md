# **Traffic Sign Recognition** 
## Kevin Harrilal Project Submission
## Writeup Template

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: /writeup_images/14_training_examples.PNG?raw=true "Data Size"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

![alt text](/writeup_images/14_training_examples.PNG?raw=true)

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It's a subplot showing some of the unmodified traffic signs, pulled at random from
each category, and the title specifies its label. 

![alt text](/writeup_images/01_unmodified_traffic_Signs.PNG?raw=true)

IF we plot the histogram of the trainning set, vs the number of categories, we can see how many images in the trainingset are classified for each label. The histogram is shown below. As we can see the data is very skewed, some labels have thousands of signs, some only have a few hundered. 

![alt text](/writeup_images/04_number_of_datapoints.PNG?raw=true)

###Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because this was used in the LeNet architecture which much success. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text](/writeup_images/02_pre_processed.PNG?raw=true)

I also attempted to normalize the data:
![alt text](/writeup_images/03_normalized.PNG?raw=true)

After, reviewing the histogram, I also attempted to generate additional data, to make up for everywhere there were less than 500 points to a particular category of signs. 

To add more data to the the data set, I took current images in the set, and randomly brightened the images, and added it to the set.
The algorithm actually selects 50 images within a single class, augments the brightness and adds it to the list. The the process repeats where 50 new images are pulled and brighted, sometimes an image can be brightend multipkle times. 

The brightness of the image is done at random and can be made darker or lighter within limits. 

Here is an example of an original image and an augmented image:

![alt text](/writeup_images/05_modify_brightness.PNG?raw=true)

The difference between the original data set and the augmented data set is the following ... 

![alt text](/writeup_images/06_number_of_datapoints_augmented.PNG?raw=true)

We can see from the updated historgram, that there are at least 500 images in each class now. 

And the final data looks like this:

![alt text](/writeup_images/15_augmented_data.PNG?raw=true)

As can be learned from the model tuning process, the final model actually removes the normalization, and data_Augmentation, and only keeps grayscaling, as that was the combination that yielded the best resuls. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The model was based off the LeNet Lab, with an addional dropout layer. 
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution  5x5   	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution  5x5   	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten | /Output 5x5x16 to 400 |
| Fully connected		| input 400 oyput 120      									|
| RELU					|												|
| Fully connected		| input 120 output 84     									|
| RELU					|												|
| Dropout					|												|
| Fully connected		| input 84 output 43    									| 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

An interative and manual process was used to train the model, the final paramerters are:

| Parameter         		|     Value	        					| 
|:---------------------:|:---------------------------------------------:| 
| Epoc | 60 |
| Batch Size | 180 |
|Learning Rate | 0.001 |
|Drop Out | 0.75 |
| mu | 0 |
|Sigma | 0.1 |
| Images | Grayscaled |


The below explains how these parameters were discovered

![alt text](/writeup_images/08_model_tuning_process.PNG?raw=true)

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 94.4%
* test set accuracy of 91.5%

![alt text](/writeup_images/07_epoch_list.PNG?raw=true)
![alt text](/writeup_images/09_test_accuracy.PNG?raw=true)

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? 
  The first architecture tried was the LeNet architecture because it was recommended by the lectures, already had a lot of experience with it in the lenet lab, and it was easier to understand, than the paper provided which highligted a modification to LeNet specifically for traffic signs. 
* What were some problems with the initial architecture? 
 The initial model only had a 86% accuracy on the validation set. I wouldn't say there was anything with the model itself, but rather with the dataset, the dataset was skewed, and not very large 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
A dropout layer was added to the initial architecture, I thought it would help improve accuracy, and it did. When dropout was added, and the batch and augmented data adjusted, the accuracy went from 89% to 93%. 
* Which parameters were tuned? How were they adjusted and why? Batch Size, Drop Out Probability, learning rate, EPOCH, and image pre-processing were all modified as described previously. They were adjusted because each had an effect in the accuracy of the model on the validation set. Sometimes parameters were adjusted, then returned to a nominal value after another paramter was adjusted because accuracy went down. Each parameter was turned making educated guesses about how the final model would be effected, and the resuling model performed particually well.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model? Some of the important design choices are the dual conolutional layers, and the many fully connected layers at the end. Dropout was helpful in preventing overfitting, which was especially useful when the data was augmented with many similar images of items in the dataset already. 

If a well known architecture was chosen:
* What architecture was chosen? LeNet
* Why did you believe it would be relevant to the traffic sign application? It is a CNN which was well understood, and a paper provided in class showed it could perform well on this problem with little modification. It was also a easy starting point since we just finished the LetNEt Lab
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?I would say 90% accuracy is pretty significant, and I have read articles of similar networks achieving 97%+. In my model I was only able to achive 94.4% accuracy on the validation set, and 91% on the test set. 
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text](/writeup_images/10_internet_images.PNG?raw=true)

The first image might be difficult to classify because an exact match doesn't exit in the dataset. This sign has what appears to be two deers on it, while the "Wild Life" traffic sign in the dataset only has 1. None the less, it was cool to try.

The second and 9th images may also be hard to classify because the 2nd image also does not have a direct match, and the 9th image has a lot more road than sign.

I tried to find the hardest images, not the easiest ones, relizing the model could probbaly perform well on esy images, I was trying to throw some curveballs. 

The data was reshaped to be 32x32x1 and then grayscaled to fit into our model. The results look like this:

![alt text](/writeup_images/11_internet_images_augmented.PNG?raw=true)


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text](/writeup_images/12_internet_images_prediction.PNG?raw=true)

The model was able to correctly guess 3 of the 10 traffic signs, which gives an accuracy of 30%. This doesn't really match the validation set, however, I would say that at least 3 of the images were extremely hard for the model to characterize. In which case. 3 out of 7 is more like ~42%. Also interesting to note, while one speed limit sign did not get the right answer, it was extrmely close.  In the next section, looking at the softmax output we explain a bit more as to where the model could have gotten hung up. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


![alt text](/writeup_images/13_internet_images_softmax.PNG?raw=true)


Some observations: 

Image 1: all predictions had a similar triagular shape. The tricky part was the icon in the center. As mentioned this sign did not have an exect replica, but the predictor was 80% certain it was a dangerous curve to the right, and all other predictions had a triangular shape, very impressive

Image 2: 97% correct prediction even though there was no exactly replica. Very impressive

Image 3: Complete failure, did not guess anything near the image. 

Image 4: Did not guess correctly, but all guess were Speed signs which was impressive, it just wasnt the right
speed

Image 5: 100% guess in correctly. Not sure how the model gets a 100% prediction. Must be something going on under the hood that just isnt right. Although the sign was round. If it is 100%, where do categories 2 and 3 come from?

Image 6: complete failure

Image 7: 100% guess correctly

Image 8: complete failure

Image 9: It was a speed sign, but the top guess does look like the sign, because it was so streched and skewed, this was another hard one to classify

Image 10: 100% corect

Overall, for my first real Deep Network, It is pretty cool to see it guess even 1 image right, without having written any code for it to do so. This is cool stuff. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


