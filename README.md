# ChatGPT Sentiment Analysis 

Twitter, a prominent social media platform with around 450 million monthly users as of 2023, generates nearly 6,000 tweets per second in 2022. The majority of Twitter users fall within the 25-34 age group, constituting approximately 38.5% of its user base. Consequently, Twitter serves as a valuable source for sentiment data collection and language model development.

With the widespread popularity of ChatGPT, an AI chatbot from OpenAI, public discussions about it have surged. This article aims to conduct sentiment analysis and construct natural language processing models to gauge public sentiments toward ChatGPT. This analysis offers insights into areas for improvement and allows for ongoing monitoring of public opinion over time. The findings are beneficial not only for ChatGPT developers but also for businesses and individuals using AI chatbots. Leveraging public sentiment data enables us to enhance AI chatbot design and functionality, aligning them more closely with user preferences and requirements

# Dataset Description

"We utilized the 'ChatGPT Sentiment Analysis' public dataset, which was publicly available and uploaded by user CHARUNI SA in 2023. This dataset comprises 219,293 entries with three columns: Serial number, tweets (text), and labels (sentiment).
<img width="462" alt="Screenshot 2023-10-05 at 12 22 36 PM" src="https://github.com/jennyliyiyuan/ChatGPT-Sentiment-Analysis-NLP-/assets/133256378/4c63c309-e74b-4316-b649-eab29b1e7c33">

The data primarily represents Twitter activity from a recent month, focusing on users' comments and opinions about ChatGPT. The 'tweets' column contains unprocessed tweets, while the 'labels' column indicates the sentiment of each tweet, categorized as 'good,' 'neutral,' or 'bad.' According to our initial analysis, approximately 49.16% of tweets had a 'bad' sentiment, 25.54% had a 'good' sentiment, and 25.30% were 'neutral.' More details can be found in Figure 2.

<img width="386" alt="Screenshot 2023-10-05 at 12 24 33 PM" src="https://github.com/jennyliyiyuan/ChatGPT-Sentiment-Analysis-NLP-/assets/133256378/da4f58fe-9db4-484c-917e-6d43cf5c8b18">

<img width="378" alt="Screenshot 2023-10-05 at 12 26 00 PM" src="https://github.com/jennyliyiyuan/ChatGPT-Sentiment-Analysis-NLP-/assets/133256378/f5cedb64-1ccb-40de-8cde-45d8077f7a85">

To enhance the dataset, we conducted data cleaning by removing null values, specific words like 'ChatGPT,' 'chatgpt,' 'OpenAI,' and URLs from the tweets.
We also generated word clouds for each sentiment category ('good,' 'bad,' 'neutral') and an overall word cloud to identify the most frequent keywords. In Figure 3, the most prevalent terms across all categories include 'AI,' 'write,' 'asking,' 'using,' and 'know.' In the 'good' category, frequent keywords include 'AI,' 'asked,' 'use,' 'good,' and 'help.' Similarly, the 'bad' category features keywords like 'AI,' 'write,' 'will,' 'asked,' and 'using,' while the 'neutral' category contains terms such as 'AI,' 'use,' 'new,' 'now,' and 'Google.' Overall, the word clouds exhibit remarkable similarities, emphasizing the focus of Twitter discussions on AI capabilities."

<img width="334" alt="Screenshot 2023-10-05 at 12 26 39 PM" src="https://github.com/jennyliyiyuan/ChatGPT-Sentiment-Analysis-NLP-/assets/133256378/ebabf562-80b8-4989-8bcc-6544bd597f44">

<img width="330" alt="Screenshot 2023-10-05 at 12 27 06 PM" src="https://github.com/jennyliyiyuan/ChatGPT-Sentiment-Analysis-NLP-/assets/133256378/1f2c1f4b-0419-46e0-a1c7-caf695091787">

<img width="345" alt="Screenshot 2023-10-05 at 12 27 24 PM" src="https://github.com/jennyliyiyuan/ChatGPT-Sentiment-Analysis-NLP-/assets/133256378/11fdff2c-e915-40c7-a7d1-b4dab6afcec1">

<img width="370" alt="Screenshot 2023-10-05 at 12 27 43 PM" src="https://github.com/jennyliyiyuan/ChatGPT-Sentiment-Analysis-NLP-/assets/133256378/be79cf85-81ea-4013-8274-f4f543ac5bb8">


# Approaches
This dataset mainly contains 3 columns, and we want to discover the sentiments of each tweet and develop a reliable and accurate classification model to help predict the sentiment of given tweets. Thus, it is necessary for us to clean and preprocess the dataset in order to better fit the models we want to build. In this project, we processed the tweet data using a number of methods and created a new dataset with the useless information and data removed.

## 3.1 Missing Values
We downloaded the data from the Kaggle website and we could not make sure whether the dataset was clean or not, we decided to check if there was any null or missing value in this dataset. After the screening, as shown in Figure 4, the dataset was clean and there was no missing value or null value existed in this dataset.

<img width="347" alt="Screenshot 2023-10-05 at 12 29 53 PM" src="https://github.com/jennyliyiyuan/ChatGPT-Sentiment-Analysis-NLP-/assets/133256378/78bb75b1-811f-4f0b-89ad-ff6480ad2b50">

## 3.2 Text Preprocessing
Before we conduct text vectorization and build NLP models, we need to preprocess the tweets in a standard way to make the texts format better for the future process. Specifically, we tokenized all the texts and removed all the punctuation, special characters, and English stop words, and we also
transformed all uppercase into lowercase. Then, we lemmatized words to their root format.

<img width="412" alt="Screenshot 2023-10-05 at 12 30 40 PM" src="https://github.com/jennyliyiyuan/ChatGPT-Sentiment-Analysis-NLP-/assets/133256378/6d296c00-b5f2-4a40-9b43-f74faa86239d">

## 3.3 Text Vectorization
After text preprocessing, we need to convert the text data into a normalized numerical form, which is also referred to as vectors, so that they can be easily used to build models. In this project, we mainly used three types of text representation methods, which are Bag of Words (BoW), Term Frequency-Inverse Document Frequency (TF- IDF), and Word Embeddings such as Word2Vec. We will discuss the vectorization methods in detail in the experiment section of this article.

## 3.4 Model Training
For model training in this project, we primarily employed logistic regression, a statistical model commonly used for predictive analysis and classification. Logistic regression estimates the likelihood of an event occurring. Given that we have three different sentiment categories with no specific order, we opted for multinomial logistic regression.

Additionally, we utilized the LDA (Linear Discriminant Analysis) algorithm, which is a popular machine learning method for classifying samples based on input features. LDA is a probabilistic model that estimates the probability of a sample belonging to a specific class. It achieves this by projecting data onto a lower-dimensional space and aims to find the best linear combination of features to separate classes effectively. To accommodate our three sentiment categories without a specific order, we applied LDA with the multinomial option, enabling classification without assuming any particular order among the categories.

## 3.5 Model Evaluation
The goal of this project is to build reliable and accurate models to help computer scientists monitor the sentiment of the public toward ChatGPT and help them make decisions in response to public sentiment. Thus, after training the model, we have to evaluate the performance of the models we made and compare them to find out the model that has the best performance.
There are many model evaluation metrics we can use, such as accuracy, precision, recall, micro F1 score, macro F1 score, and Area Under the Receiver Operating Characteristic Curve (AUC- ROC).
Accuracy measures the percentage of true positives and true negatives in the dataset. Precision is a metric that focuses on the correct prediction of the positive class while recall is a metric that focuses on identifying true positives in the dataset. F1 scores consider both precision and recall and thus provide a more balanced way to measure the performance of models. AUC-ROC is a measurement for classification problems and ROC is a probability curve and AUC is the degree of separability. Usually, the higher the scores, the higher the performance. In this project, we mainly used F1 scores as our measurement metrics since they are more balanced in measuring our models, which is particularly important in sentiment analysis where the consequences of misclassifying sentiment can vary in significance. F1 scores provide a more meaningful measure of how well the models perform in identifying both positive and negative sentiments, which is crucial for the project's objective of helping decision-makers respond to public sentiment.

# Experiment

## 4.1 Zip'f Law

We tested Zipf's law based on a corpus of text data. We first imported the necessary libraries, including defaultdict from the collections module, re for regular expressions, and matplotlib for plotting. It then created a corpus variable by joining all the tweets from a data frame, removes all non- alphanumeric characters from the corpus using regular expressions, and converts all words to lowercase.

Next, we code to create a defaultdict object called “freq”, which will be used to count the frequency of each word in the corpus. It then iterates over each word in the corpus, updating the count in the freq dictionary for each word. The resulting freq dictionary is sorted in descending order based on the frequency of each word, and the top 100 words are printed.
Finally, we plot the frequency distribution of the words using a log-log plot. The x-axis represents the word's rank in the frequency table, and the y- axis represents its frequency. The resulting plot can be used to visualize how well Zipf's law holds for the given corpus of text data.

The results show the top 100 words in the corpus, along with their frequencies. The most common word in the corpus is 'chatgpt', which appears 229,140 times, followed by 't', 'co', 'https', and 'the'.

<img width="400" alt="Screenshot 2023-10-05 at 12 48 58 PM" src="https://github.com/jennyliyiyuan/ChatGPT-Sentiment-Analysis-NLP-/assets/133256378/8bd0f302-0176-4129-8299-b895bf239586">

## 4.2 Binary Vectorizer
First, we decided to use Binary-valued vector processing to convert textual data into binary vectors. This process is an important step of text feature engineering. The function if_exist_vocab takes a document and a vocabulary as inputs and returns a binary-valued vector that represents the presence or absence of each word in the vocabulary in the input document. The resulting binary vector can then be used as input to machine learning models such as logistic regression for classification.

The first step in the preprocessing pipeline is tokenization, which involves splitting the input document into individual words or tokens. We use the pre_processing_by_nltk function for tokenization. After tokenization, we iterate over each token in the document and check whether it is present in the vocabulary. If the token is not in the vocabulary, it is replaced with a special token <UNK> to indicate that it is an unknown word. Then, the binary vector has a 1 for each word in the vocabulary that is present in the document and a 0 for each word that is not present.

The resulting binary vector can be used as input to machine learning models such as logistic regression for classification. We use the Logistic Regression model to train on the binary vectors using the fit method, and its performance is evaluated using the AUC-ROC score and the micro and macro F1 scores.

The value of AUC-ROC is 0.9305, which indicates that the classifier is quite good at distinguishing between the different categories. The value of Micro F-1 ranges from 0 to 1, with a higher value indicating better performance. In this case, the value of Micro F-1 is 0.8371, which indicates that the classifier is quite accurate in predicting both the positive and negative classes. The value of Macro F-1 is 0.8139, which indicates that the classifier is quite good at predicting each class separately.

We also implemented the LDA algorithm to predict the target variable for a given set of features. We first created an instance of the LDA model and then fitted the model on the training data using the fit() method. After training, the model is used to predict the target variable for the test data using the predict() method. The predicted values are stored in y_pred.

The accuracy of the model is found to be 0.805, which indicates that the model correctly predicted 80.5% of the test samples. The F1 score is found to be 0.776, which suggests that the model has a good balance between precision and recall. The Recall score is found to be 0.765, which indicates that the model has a good ability to detect positive samples.

## 4.3 Frequency Vectorizer
We used tf() function takes a document (tweet) and a vocabulary as inputs. It first tokenized the document using a pre-processing function from the Natural Language Toolkit (NLTK) library. It then replaced any tokens not in the vocabulary with the <UNK> token (representing unknown words). Finally, it created a frequency vector of length equal to the size of the vocabulary, where each entry corresponds to the frequency of the corresponding word in the document.

The X list is initialized as an empty list, and then filled with the frequency vectors for each tweet in the dataset using the tf() function. The train_test_split() function from the scikit-learn library is used to split the dataset into training and testing sets, with 20% of the data being used for testing. The logistic regression model is trained on the training set using LogisticRegression().fit(), and its performance is evaluated on the testing set using the area under the ROC curve (roc_auc_score()) and the micro and macro F1 scores (f1_score()).
The resulting frequency vectors have a length equal to the size of the vocabulary, which is specified by the user. In this case, it appears that the vocabulary contains 400 words. Therefore, each frequency vector represents the frequency of each of the 400 words in the corresponding tweet.

The value of the AUC-ROC score is 0.9209, which indicates that the model has a good ability to distinguish between the classes. The Micro F-1 score of 0.8188 indicates that the model has a reasonably good performance on the test set. The Macro F-1 score of 0.7924 indicates that the model's performance is slightly worse when considering each class separately.

We also implemented the LDA model on a text classification problem using the frequency vectorizer. The frequency vectorizer converts the text data into numerical representation based on the frequency of words present in the text. The LDA model is then trained on the training data and used to predict the labels of the test data.

The accuracy of the LDA model trained on frequency vectorized data was 0.805, indicating that the model correctly classified 80.5% of the test data. The F1 score of the model was 0.776, which is a weighted average of precision and recall, and a measure of the model's overall performance. The recall score, which measures the proportion of actual positives that are correctly identified by the model, was 0.765.

## 4.4 TF-IDF Vectorizer
We also import the TfidfVectorizer from the scikit- learn library to transform text data into a numerical form for machine learning algorithms. The vectorizer is configured to remove accents, convert all text to lowercase, and tokenize the text using a pre-processing function defined in a separate module (pre_processing_by_nltk). The vectorizer is set to use inverse document frequency (IDF) weighting, which is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The normalization parameter is set to l2, which normalizes the vector by the Euclidean norm. Smooth_idf parameter is set to True which prevents division by zero error in case a token is not present in a document. Next, we train a Logistic Regression model on the vectorized features using the fit() method, with the input X_train and y_train data.

The performance of the trained model is evaluated on the test data, with the AUC-ROC, Micro F-1 score, and Macro F-1 score. The model achieved an AUC-ROC of 0.93, indicating a good performance. The Micro and Macro F-1 scores are 0.81 and 0.79 respectively, which means the model is fairly good at predicting the correct classes on the test data.

The AUC-ROC score is 0.932, which suggests that the classifier is quite good at distinguishing between the classes. The micro F-1 score is 0.817, which suggests that the classifier performs reasonably well on the test data. The macro F-1 score in this case is 0.787, which suggests that the classifier is good at identifying some classes but not so good at identifying others.

We also used the LDA model that is fitted to the training data using the tf-idf vectorizer. The accuracy of the model is 0.810, which indicates that the model can classify text data with high accuracy. The F1 score of the model is 0.777, which indicates that the model can achieve a balance between precision and recall. The recall score of the model is 0.769, which indicates that the model can identify a high proportion of true positives.

## Word2Vec
We applied the powerful Word2Vec model to produce word embeddings for text classification. Firstly, we split the data from our CSV file into training and testing sets and preprocessed it by removing stopwords and punctuation with the preprocess_df() function. Next, we used the word_tokenize() function from the NLTK library to tokenize the preprocessed training data.

To build a vocabulary from the tokenized training data, we employed the build_vocab() function. This function calculated the frequency of each word in the training data and mapped each word to an index. Using the get_embeddings() function, we then generated word embeddings from the training data, which leveraged the Word2Vec algorithm to learn high-quality word embeddings.

Furthermore, we utilized the word embeddings to represent each document in the training and testing data as a vector. We calculated the vectors by taking the average of the embeddings of the words in the document.

To train the model, we employed the widely- used LogisticRegression() function from the scikit- learn library and optimized the hyperparameters of the model using grid search. We evaluated the accuracy of the model on the testing data.

By utilizing the Word2Vec algorithm, we successfully generated high-quality word embeddings and trained a Logistic Regression model that achieved an accuracy of 79% on the testing data. This demonstrates the effectiveness of Word2Vec for text classification tasks and provides a promising avenue for future research in the field.

The precision, recall, and F1-score metrics are reported for three classes: 'bad', 'good', and 'neutral'. The support column represents the number of instances for each class. The overall accuracy of the model is reported as 0.79, which means that it correctly predicted the class labels for around 79% of the instances. However, it is important to look at the individual class metrics to get a better understanding of the model's performance.

Looking at the precision scores, the model performs best for the 'bad' class with a precision score of 0.85, meaning that 85% of instances predicted as 'bad' were 'bad'. The precision score for the 'good' class is 0.73, and for the 'neutral' class is 0.72.

The recall score, which represents the proportion of true positives correctly identified, is highest for the 'bad' class with a score of 0.91. The recall score for the 'good' class is 0.81, and for the 'neutral' class is 0.53.

The F1-score, which is the harmonic mean of precision and recall, is also highest for the 'bad' class with a score of 0.88. The F1-score for the 'good' class is 0.77, and for the 'neutral' class is 0.61.

The model seems to perform well for the 'bad' and 'good' classes, with F1-scores of 0.88 and 0.77 respectively. However, the model performs relatively poorly for the 'neutral' class with an F1- score of 0.61. The lower recall score for the 'neutral' class indicates that the model is having difficulty correctly identifying instances of this class. It is also worth noting that the weighted average F1- score is 0.78, which suggests the model performs well across all classes when considering the class distribution. In conclusion, the output suggests that the model trained using word2vec embeddings performs well for the 'bad' and 'good' classes but has difficulty correctly identifying instances of the 'neutral' class.

# Conclusions
This article discussed the analysis of the sentiment of people towards ChatGPT using sentiment analysis and natural language processing models. The analysis is based on a dataset collected from Twitter, containing 219,293 tweets related to ChatGPT. The report also included testing Zipf's law based on a corpus of text data and used BoW and Word Embeddings to convert the words into vectors. We mainly used logistic regression and LDA to build our classification models, and from the seven models we built, we found that the logistic regression model using binary vectorizer is the most accurate based on the F1 score.

<img width="401" alt="Screenshot 2023-10-05 at 1 04 36 PM" src="https://github.com/jennyliyiyuan/ChatGPT-Sentiment-Analysis-NLP-/assets/133256378/2f101331-7a1d-451d-a49c-36989d485a70">
