# Reddit_Sentiment_Analysis-_Classifying_Jokes_vs_Questions
## Executive summary

**Problem Statement:** Can we get a robot to tell the difference between someone telling a joke or asking a genuine question? More specifically, can we **create a classification model that can correctly distinguish the difference between a posted question from one of two subreddits: Dad Jokes and ELI5 (explain like I'm five)?** The motivation behind this project is to better understand how we can create bots that filter through large numbers of posts in comment sections or product reviews in order to quickly identify genuine questions rather than sarcastic or funny comments (such as rhetorical questions from yelp reviews). Basically, we want robots to understand humor! 

This tasks performed in this notebook include:
-	Scraping Reddit API from two separate subreddits
-	Utilizing NLP methods
-	Modeling binary classification with several different ML models.

**Data Collection:** The two subreddits were chosen based on their similarities in structure/format but also containing dissimilar subject topics. A total of 3259 posts were scraped from both subreddits using simple Reddit API methods to download and convert html to json files in under ~ 1 hour. The json files were converted to a dataframe, posts were cleaned, removed of any distinguishing tags, and given binary labels. 

**Pre-processing and Modeling:** Before any models could be tested, the posts were pre-processed using NLP methods. In order to determine the best text format that posts should be in for modeling, a preliminary filtering evaluation was performed that compared the accuracy scores with a logistic regression model where filtering methods were iteratively tested (i.e. all filtering methods down to no filtering methods).  The level of pre-processing was determined and a baseline model score using logistic regression was obtained. Using the same level of pre-processing, five other machine learning models from sk.learn were tested:
- Multinomial NB
- Random Forest Classifier
- Extra Trees Classifier
- Bagging Classifier
- Ada Boost Classifier

**Results:** Model performance was evaluated based on accuracy scores for training and test data. For each algorithm type the data was fit using the models default parameters as well as with some parameter optimization using Gridsearch. The classifier that produced the highest overall accuracy score was logistic regression. However, parameter optimization did not improve the models performance from the baseline, wheras almost every other model improved with parameter optimization. Between the five other classification models tested, Extra Trees Classifier produced the highest accuracy score. With the exception of Ada Boost, all models suffered from overfitting and could further be optimized with a focus on reducing variance.

---

## Explanation of all files and directories in repo
<br>This repo contains:

- **Notebooks:**
	- __[1a-reddit_data_collection_DAD_JOKES.ipynb](https://git.generalassemb.ly/amytaylor/Binary-Classification-using-Reddit-API-Jokes-vs.-Questions/blob/master/notebooks/1a-reddit_data_collection_DAD_JOKES.ipynb)__ Contains data collection and storage for the Dad Jokes subreddit
	- __[1b-reddit_data_collection_ELI5.ipynb](https://git.generalassemb.ly/amytaylor/Binary-Classification-using-Reddit-API-Jokes-vs.-Questions/blob/master/notebooks/1b-reddit_data_collection_ELI5.ipynb):__ Contains data collection and storage for the ELI5 subreddit
	- __[2-Data_Cleaning_dad_eli5.ipynb](https://git.generalassemb.ly/amytaylor/Binary-Classification-using-Reddit-API-Jokes-vs.-Questions/blob/master/notebooks/2-Data_Cleaning_dad_eli5.ipynb):__ Both dataframes combined, cleaned, and EDA performed.
	- __[3-Model_Evaluation.ipynb](https://git.generalassemb.ly/amytaylor/project3_reddit/blob/master/notebooks/3-Model_Evaluation.ipynb):__ Classification model evaluation

- **Datasets:**
	- [dad.csv](https://git.generalassemb.ly/amytaylor/Binary-Classification-using-Reddit-API-Jokes-vs.-Questions/blob/master/datasets/dad.csv) : 1738 unique posts downloaded as json from 'Dad Jokes' subreddit and converted to dataframe containing ten features. Created in Notebook-1a, used in Notebook-2.
	- [eli5.csv](https://git.generalassemb.ly/amytaylor/Binary-Classification-using-Reddit-API-Jokes-vs.-Questions/blob/master/datasets/eli5.csv) : 1521 unique posts downloaded as json from 'ELI5' subreddit and converted to dataframe containing ten features. Created in Notebook-1b, used in Notebook-2.
	- [dad_five.csv](https://git.generalassemb.ly/amytaylor/Binary-Classification-using-Reddit-API-Jokes-vs.-Questions/blob/master/datasets/dad_five.csv) : Dataframe of both subreddits combined, cleaned and narrowed down to two features (text and target classification) Created in Notbook-2, used in Notebook-3.
	
- **Data Dictionary of all features in dad.csv and eli5.csv:**
<br>_all features used for minimal EDA purposes_

| Feature | Type | Dataset| Description |
| ----| ---- | ---- | ---- |
| `name`   | object | dad_five.csv and eli5.csv |  unique user id per post |
| `title`   | object | dad_five.csv and eli5.csv|  actual posted text; contains majority of full post |
| `selftext `   | object | dad_five.csv and eli5.csv | additional text posted; empty for most submissions  |
| `subreddit`   | object | dad_five.csv and eli5.csv |  name of subreddit that post comes from |
| `created`   | float | dad_five.csv and eli5.csv | date created  |
| `author`   | object | dad_five.csv and eli5.csv |  reddit users name |
| `num_comments`   | int | dad_five.csv and eli5.csv | number of comments received per post  |
| `ups`   | int | dad_five.csv and eli5.csv | number of up votes per post  |
| `downs `   | int | dad_five.csv and eli5.csv | number of down votes per post  |
| `score`   | int | dad_five.csv and eli5.csv | total score = summation of up and down votes  |

- **Data Dictionary of features in dad_five.csv used for modeling in Notebook-3 :**    

| Feature | Type | Dataset| Description |
| ----| ---- | ---- | ---- |
| `post`   | object | dad_five.csv |  unique post (`title` and `selftext` combined and cleaned) |
| `subreddit`   | int | dad_five.csv |  binary column for subreddit class: 0= dad jokes, 1 = ELI5 |

---
# Project Summary
### 1. Data Collection and Storage
<br>_The process for scraping posts from the Reddit webpages can be found in Notebook-1a (dadjokes) and Notebook-1b (ELI5)_
	
**Choice of subreddits:** The two subreddits were chosen based on similar format structure: the majority of posts in both categories are typically 1-2 sentences in length, they contain a variety of topics, and most are framed as questions. While almost every single ElI5 post is framed with typical question syntax (contains words: ‘how’, ‘what’, etc), only about half of the Dad Jokes post are posed as questions. What also makes them good candidates for comparison is  the broad variety of subjects discussed. This absence of central themes means that the model can (hopefully) focus on distinguishing questions from jokes by focusing on order of words, stopwords, or other non-obvious syntax rather than specific words that might be common in a more specific subreddit (aka r/AskScience, etc). 

Here is an example of 5 posts from each category:

| example posts from r/dadjokes | example posts from r/eli5 | 
| ---- | ---- |
| What kind of exercise do lazy people do? Diddly squats| Why don’t you hear about the history of Africa before colonialism? |
| Why did the snowman name his dog Frost? Because sometimes Frost bites! | Why do animals seem to have a hard time seeing themselves in a mirror? |
| What's the difference between a depression and a recession? A recession is when you lose your job, a depression is what happens when I lose mine. | What is the difference between movies and films? Also directors and filmmakers? |
| A ghost either is or is not resting against a wall. boolean|Gamma regression and when it should be used. |
| What do you call it when the Indian restaurant forgets your bread? It’s a naan issue.| Why is the body able to regrow the liver and not any other organs? |


**Data collection method:** Both subreddits were scraped using easy scraping methods from the Reddit API. A total of 3259 posts were collected (1738 from Dad Jokes and 1521 from ELI5). 

Posts were obtained from the following four locations:
  - https://www.reddit.com/r/dadjokes/top/?t=month
  - https://www.reddit.com/r/dadjokes/new/
  - https://www.reddit.com/r/explainlikeimfive
  - https://www.reddit.com/r/explainlikeimfive/controversial




### 2.Cleaning and EDA
<br>_All data cleaning and EDA tasks can be found in Notebook-2_

The following cleaning tasks were performed:
 - Replacement of null values and removal of discriminatory tags such as "ELI5"
 - Combined comments from two columns into one master "posts" column
 - Binarized subreddit category

Some EDA findings of submission qualties include:
- i. Analysis of posts with the most: author submissions, comments, and scores
- ii. Findings (not surprisingly):
	- r/dadjokes posts have significantly higher scores, or more "likes", while r/eli5 posts have significantly more comments and discussion
	- much of r/dadjokes posts are written by repeat authors, whereas r/eli5 posts are not: the same ten people contributed 14% of all posts in the corpus, whereas the top ten people contributed only 1.7% of total posts, respectively.

Other EDA findings of word counts from each post: (red outline indicates unique words in both subreddits)

![Alt text](https://github.com/amytaylor330/Reddit_Sentiment_Analysis-_Classifying_Jokes_vs_Questions/blob/master/images/stopwords.png)
---
![Alt text](https://github.com/amytaylor330/Reddit_Sentiment_Analysis-_Classifying_Jokes_vs_Questions/blob/master/images/no_stopwords.png)



### 3.Modeling.  (Model performance on training/test data)
<br>_All modeling and evaluation tasks can be found in Notebook-3_ 

**i. Baseline Model**
<br>To determine the best text format that posts should be in for modeling, a preliminary filtering evaluation was performed that compared the accuracy scores with a logistic regression model where filtering methods were iteratively tested (i.e. all filtering methods down to no filtering methods).  The level of pre-processing was determined and a baseline model score using logistic regression was obtained. Results summarized in the following table:
![Alt text](https://github.com/amytaylor330/Reddit_Sentiment_Analysis-_Classifying_Jokes_vs_Questions/blob/master/images/preprocessing.png)

In order of importance, these results show that: stopwords must be included, removing HTML tags helps, lemmatizing words does not matter (and could be omitted), and removing punctuation/digits has minimal to no effect. 
Based on these results, the decision was made to use the automatic pre-processing features in CountVectorizer for all models.

**ii. Model Comparison**
<br>Using gridsearch pipeline, six classification models were evaluated for accuracy using each models' default parameters as well as with some gridsearch parameter optimization.
The results are summarized in the following table:
![Alt text](https://github.com/amytaylor330/Reddit_Sentiment_Analysis-_Classifying_Jokes_vs_Questions/blob/master/images/results.png)

While logistic regression performed the best overall, it was not much improved with parameter optimization. All models (except for Ada Boost) suffered from overfitting and could further be optimized with a focus on reducing variance. The second best score was produced by Extra Trees Classifier, which introduces more randomness than Bagging Classifier and Random Forest Classifier, but still managed to produce less bias. 
    

### 4.Further Actions

If I had more time to explore this project further I would:
-	Experiment with creating a unique set of stopwords to try generalize the model. For example, the word “dad” occurs occasionally in the Dad Jokes thread; if this word and others like it were removed, perhaps the model could focus on less discriminating features, or avoid overfitting. Also, since stopwords were included for the model’s best performance, it would be nice to evaluate how removing a shorter set of stopwords affect its performance, i.e. removing “and”, “is”, “the” but keeping important words like “why”, “how”, etc. 
-	Focus on reducing overfitting. I would start by removing features included in the model. 
-	Re-try processing parameters with decision tree classifiers
