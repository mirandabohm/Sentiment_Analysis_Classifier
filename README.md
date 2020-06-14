# A Sentiment Classifier Built in Keras and Trained on Kaggle Twitter Airline Sentiment Data 
  
## Getting Started
After downloading this repo, only the run_me.py file needs to be directly run. 

Model build and training occurs in RNN.py. There, parameters can be tweaked and performance can be evaluated. File includes matplotlib visualizations of training and testing accuracy and loss.  

As the goal was to evaluate the overall sentiment of a string as 'positive', 'negative', or 'neutral', this task has been framed as a three-class classification problem. Thus, the output of the top Dense layer is 3. Alternatively, a regression approach could have been used.   

Due to limitation of computational resources and time, I opted to use a pre-trained set of word vectors trained on a very large corpus of Twitter data. GloVe is a an unsupervised solution derived from the distributional hypothesis (“words which are similar in meaning occur in similar contexts.” (Herbert Rubenstein and John B. Goodenough. 1965.
Contextual correlates of synonymy. Communications of the ACM). 

With early stopping, training ceased after 37 epochs and yielded the following success metrics: 

accuracy: 0.81831
recall: 0.80704
precision: 0.82999

```
batch_size = ~64
num_epochs = ~35
LSTM units = ~32 # 15 to 30 between two or more layers; higher numbers provoked overfitting
Dropout = ~0.20
``` 

![Image of Accuracy vs. Epochs](Images/Figure_9.png)![Image of Accuracy vs. Epochs](Images/Figure_10.png)


![Image of Loss vs. Epochs](Images/Figure_11.png)![Image of Loss vs. Epochs](Images/Figure_12.png)

## Dependencies
GloVe: Global Vectors for Word Representation 
I used the 50d Twitter set found here: 
https://nlp.stanford.edu/projects/glove/

The Twitter U.S. Airline Sentiment dataset from Kaggle
Dimensions: 14640 examples by 15 features. Sentiment is 63% negative, 21% neutral, 16% positive.*

https://www.kaggle.com/crowdflower/twitter-airline-sentiment

As outlined below, such class skew can be problematic, and I hope to address this in future iterations of this project. 

Tensorflow (including Keras):  
https://www.tensorflow.org/install/pip
```
! pip install tensorflow
```

NumPy:

https://docs.scipy.org/doc/numpy/user/install.html
```
! pip install numpy
```
Matplotlib (for visualizations): 
https://matplotlib.org/users/installing.html
```
! pip install matplotlib
```

## Project Structure
The GloVe and Kaggle data will need to be downloaded from their respective sites and placed within a subdirectory named Data. 

## Future Tasks
Model accuracy averaged around 75-85% on training and testing data, with various parameters. 
While this is in range of acceptable, improvements could be made by further exploring:  

* Use of a dataset covering a larger domain, i.e. representing a greater percentage of words used in everyday English. Because this set deals primarily in language about airlines in the United States, it does not generalize well to other arenas.
* Use of a dataset with higher cohesion. Excerpts from iterary sources, for example, would include fewer typographical errors, "slang" words, emojis, and other irregularities that can confuse learning. 
* Use of a dataset without skew between classes; or, because some discrepancy is nearly inevitable, better manipulation of data to account for class imbalance. 
* More extensive data cleansing to accomodate a higher percentage of missing words, as well as digits and other outliers not addressed. 

I hope to implement RNN "from scratch" - that is, without the use of APIs, in the near future, in the same vein that I have written vanilla nets using only Numpy in the past. Since LSTM has garnered wide preference over naive ANNs, CNNs, and even RNNs, for sentiment analysis, I opted to start there. 

## Final Thoughts
This has been a significant learning experience for me, and I sincerely appreciate the opportunity.

My script was created in Spyder in Python v. 3.6.0 and Anaconda 2019.10. 
