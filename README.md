# An LSTM-Based Sentiment Classifier Built in Keras and Trained on Kaggle Twitter Airline Sentiment Data 
  
The goal was to build a classifier that could identify the overall sentiment of a user-inputted string, with three possible labels: 'positive', 'negative', and 'neutral'. Thus it was appropriate to frame this task a three-class classification problem, where every prediction yields a class label and a likelihood score. Alternatively, a regression approach could have been used. 

In order to boost accuracy and reduce  computational resources, I opted to use a pre-trained set of word vectors trained on a very large corpus of Twitter data. GloVe is a an unsupervised solution derived from the distributional hypothesis (“words which are similar in meaning occur in similar contexts.” (Herbert Rubenstein and John B. Goodenough. 1965.
Contextual correlates of synonymy. Communications of the ACM). 

## Getting Started
Version: Python v. 3.7.6 and Anaconda 4.8.3. 

## Dependencies
The GloVe and Kaggle data will need to be downloaded from their respective sites and placed within a subdirectory named Data. 

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

Download the repo and run get_glove_model.py to build glove_model.npy and avg_vec.npy. Once these files are written, run run_me.py. I will adapt project structure to automatically call get_glove_model.py from within run_me.py, going forward. 

An overview of each module: 

### RNN.py: Build and Train Model
By default, model architecture is Sequential, with two LSTM layers, and a topmost Dense layer. Editable attributes include Dense activation, number of hidden units, and Dropout regularization. Hyperparameters such as batch size, maximum allowable epochs, and early stopping characteristics can also be tweaked here. Output is a model.h5 model file which can be called directly, without need for repetitive retraining. RNN also includes performance evaluation and matplotlib visualizations of accuracy and loss.  

### run_me.py: Load .h5 Model File and Present Interactive Interface
Loads model and provides a command line for users to input sentences, which will then be judged for their sentiment. Predictions will be graded with a likelihood score. 

### glove_data.py: Builds Test and Train Datasets
First constructs a 3D array containing word vectors for every token included in every Tweet in the dataset. 

### process_text.py: Cleans Tweet Data
Includes regex expressions to lowercase, tokenize, and format Tweet data. Removes URLs, @mentions, and other junk punctuation. Does not remove stopwords, which was intentional, but may add this optional functionality later. Returns a list of lists. 

### get_glove_model.py: Load GloVe Pre-trained Word Vectors
Output is a dictionary called gloveModel, wherein keys are vocabulary tokens (strings) and values consist of pre-trained word vectors (Numpy arrays of length 50). Vector lengths greater than and less than 50 can be chosen instead on the Kaggle website. 
The separateness of this module from glove_data.py precludes the need to load the GloVe dictionary with every model run. 

### get_inputs.py: Creates a Dataset Object 
Builds a structured Tweets dataset, including cleaned data and one-hot-encoded labels. 

## Results

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

## Future Tasks
Model accuracy averaged around 75-85% on training and testing data, with various parameters. 
While this is in range of acceptable, improvements could be made by further exploring:  

* Use of a dataset covering a larger domain, i.e. representing a greater percentage of words used in everyday English. Because this set deals primarily in language about airlines in the United States, it does not generalize well to other arenas.
* Use of a dataset with higher cohesion. Excerpts from iterary sources, for example, would include fewer typographical errors, "slang" words, emojis, and other irregularities that can confuse learning. 
* Use of a dataset without skew between classes; or, because some discrepancy is nearly inevitable, better manipulation of data to account for class imbalance. 
* More extensive data cleansing to accomodate a higher percentage of missing words, as well as digits and other outliers not addressed. 

I hope to implement RNN "from scratch" - that is, without the use of APIs, in the near future, in the same vein that I have written vanilla nets using only Numpy in the past. Since LSTM has garnered wide preference over naive ANNs, CNNs, and even RNNs, for sentiment analysis, I opted to start there. 