# Project 3 - Web API's and Classification #

**********

## Problem Statement ##

Nowadays internet is full of discussion forums where billions of people freely talk about zillions of things. Quite often these discussions are a perfect source of information on political views, customer preferences, weather, ways to morgage one's grandma's dentures - just to name a few. 

Quite often we come across a situation when we need to identify a source of information for reasons such as information collection, research, source identification. etc. Or simply that poor grandma's unable to locate her dentures demands to know everything - especially since she has come across your morgaging ad drafts!

For these viable purposes we want to know where a chunk of text comes from, and this can be a tricky task. And every modern grandma wouldn't simply trust your statements without any validated research results. While many grandmas can now google more than we can imagine, and some can even scrape the web, the problem is much more complex.

Indeed, in order to identify the information sources we should often start from understanding where our sample comes from. And here our modeling techniques for NLP (Natural Language Processing) come in handy. In order to answer our grandma's questions properly we need to be able to have a model predicting the source of our information.

To start from simple things and later get to more advanced investigating techniques, let's see if we could **predict what of the two discussion threads our text sample comes from, and what classifier model works best for it.** That will immensely help a team of data scientist proceed with developing a more complex model to answer concerned grandma's questions in future, and will definitely lead this curious team to eventually winning an Ig Nobel prize accurately predicting an answer to the most thrilling question of all times "Where the```list_of_stopwords```have you been?" and, ultimately, receiving the Nobel Peace Award for eliminating the source of 99,(9)% domestic disputes.
 
In order to answer the post classification question and to determine the best suitable predictive model I eveluated performance of 10 different models and some of their variants, and tried to improve the performance of some of them seeming to be 'most promising'. As an evaluation criteria I chose to use the model's accuracy, as it reflects the idea of binary outcome prediction best.
 
**********
## Project Files ##

The project consists of two separate Jupyter Notebooks listed in the order of their workflow sequence and available at the __[code](http://localhost:8888/tree/code)__ folder in the Project repository:

1. __[Principal Code](http://localhost:8888/notebooks/code/Principal%20Code.ipynb)__
2. __[Auxillary Images](http://localhost:8888/notebooks/code/Auxillary%20Images.ipynb)__ (used solely for Project Presentation illustrations generation)

Datasets used for the Project and generated through the Project are available at the __[data](http://localhost:8888/tree/data)__ folder in the Projects repository.

Auxillary images generated for the Project in a separated Jupyter Notebook __[Auxillary Images](http://localhost:8888/notebooks/code/Auxillary%20Images.ipynb)__ are available as *.png files at the __[images](http://localhost:8888/tree/images)__ folder in the Projects repository.


**********
## Data Dictionary ##
Data in the __[data](http://localhost:8888/tree/data)__ folder consists of three *.csv files. Files 'AskMen.csv' and 'AskWomen.csv' are the intermediate results of web-scrapping from the corresponding subreddits at __[reddit.com](https://www.reddit.com)__ . Their original sources can be found at the following locations:

| Subreddit 	| Location                                        	|
|-----------	|-------------------------------------------------	|
| AskMen    	| **[Link to AskMen](https://www.reddit.com/r/AskMen)**   	|
| AskWomen  	| **[Link to AskWomen](https://www.reddit.com/r/AskWomen)** 	|

Both these files were merged into the 'data_raw.csv' file where the data features and their respective formats look like:

| Feature 	| Type   	| Description                                    	|
|---------	|--------	|------------------------------------------------	|
| text    	| object 	| Actual text of a post in one of the subreddits 	|
| source  	| object 	| Subreddit's name as data source identifier     	|



**********
## Executive Summary ##

One doesn't have to be a perfect expert in computer technoligies and Natural Language Processing to understand that differentiating between two given chunks of text from the web can be extremely tricky. 

A piece of human-written text is a highly irregular structure by itself. In order to analyze it, we must always take into attention things like punctuation, lexic patterns, highly possible typo's and formatting issues. Luckily, we could use brilliant Sklearn libraries for NLP doing a lion's share of work for us. 

Approaching the problem of subreddits identification we could use various models from the same Sklearn Python's library. It is of paramount importance to properly identify our Baseline Model beforehand. In our situation it is the rate of available posts from a larger subreddit to the total number of posts in our data base, or 272 posts from 'AskMen' thread to 404 posts in total, which equals roughly to 0.6606.

For this project I chose to start with a classical Logistic Regression model as a natural 'first choice' classifier with two vectorizer options and grid searched optimized parameters. Since it's performance identifying a given text post's subreddit with a close to 75% accuracy (0.7456 for a Logistic Regression model using Count Vectorizer, to be precise) does leave a room for imporovement, I attemted it by using other more advanced classification models from Sklearn.

The two Naive Bayes models used after (as a model choice in case with Naive Bayes models depend on the model's input data format, hence here with NLP - on a chosen vectorizer) didn't really work significantly better for our purpose with the results in the close proximity of our Logistic Regression (Multinomial NB Model exceeding LR by less than 0.5%, and Gaussian NB performing nearly 5% worse than LR). 

Then I used four models from the "Decision Tree" family: Decision Trees, Bagging Classifier, Random Forest and Extra Trees. Whenever appropriate I used model's hyperparameters optimization by grid search. It resulted with me achieving the best accuracy score for the Random Forest model with Count Vectorizer and optimized by hyperparameters grid search of 0.7700 - which is an improvement for the Logistic Regression result.

Finally, I tried using three more advanced models - AdaBooster, Gradient Boosting and Support Vector Machines. The results for these three models with Count Vectorizer and optimized by hyperparameters grid search were not of impressive improvement in comparison with the Logistic Regression, exceeding it by a very little margin for the Support Vector Machines model's accuracy score of 0.7770.

The SVM  model's result is the best across all the models I used, it is a nearly 12% prediction accuracy improvement at the Baseline Model's level of 0.6606, but just a bit more than a 2% hike in comparison with the Logistic Regression model. 


**********
## Conclusions and Recommendations ##

### Conclusions: ###
My best model has turned out to be the Support Vectors Machine, and this is a expected result. This model has shown the highest accuracy on a testing set, and roughly 3 out of 4 internet posts could be correctly identified as belonging to one of the two threads.
Quite an unexpected part of this result is that the model's performance is quite insignificantly better (by just 2%) than the model's of the first choice - Logistic Regression.

    
### Recommendations: ###
Considering the current project I'd suggest the following:
- Working with bigger dataset, as current entire datasets contained just 383 posts
- Experiment with more optimization options - grid searching through more hyperparameters while paying attention to extending/narrowing each parameter's range and iteration steps through the corresponding ranges. _This inflicts quite some computational capacity problems_

Generally speaking, if we wanted to significantly improve our model's accuracy classifying pieces of human-produced text we should definitely consider using more advanced algorithms.

**********
## Source Documentation ##

Images used in Project Presentation are sourced at **[Wikimedia Commons](https://commons.wikimedia.org/wiki/Main_Page/)** and are copyright-free.


# NLP-project
# NLP-project
