# CSC380 Principles of Data Science Final Project
# Using Network Attack Data to Classify Attack Types Using Machine Learning.

**Outline**

  - [Objective](#Objective)
  - [Data](#dataset)
  - [Data Exploration](#Data-Exploration-and-Insights) 
  - [Data Preprocessing](#Data-Preprocessing)
  - [Feature Selection/Engineering](#Feature-Selection/Engineering)
  - [Model Selection](#Model-Selection)
  - [Model Training and Evaluation](#Model-Training-and-Evaluation)
  - [Model Selection](#Model-Selection)
  - [Results and Conclusion](#Results-and-Conclusion)
  - [Limitations](#Limitations)
## Objective

The goal of my data science project was to predict the type of network attack given 42 network variables out of 44 features (label is binary and represents benign or non benign, I removed this feature as not to bias the model). I used 3 machine learning models to categorize and predict the attack type. I trained each of them on various subsets of the total data set. This included the full set, only attack data, outlying value removed, optimized feature reduced subsets, and a few more. 

## Dataset
I chose NF-UNSW-NB15-v2 which is the network flow data from network intrusion detection systems at https://staff.itee.uq.edu.au/marius/NIDS_datasets/#RA6
A description is here https://research.unsw.edu.au/projects/unsw-nb15-dataset

The dataset include features such as:
    
    Source IP address
    Source port number,
    Destination IP address,
    Destination port number,
    Transaction protocol,
    Source to destination transaction bytes, 
    Source bits per second,
    Destination bits per second,
    Source to destination packet count, 
    Destination to source packet count,
    and more.
    
There are 10 types of attacks I consider:
['Fuzzers', 'Benign', 'Exploits', 'DoS', 'Reconnaissance', 'Generic', 'Shellcode', 'Analysis', 'Backdoor', 'Worms']. Benign is actually the lack of any attack and represents the majority of the data set. The 44 features that represent network data are listed at the link above.


## Data Exploration and Insights

Of the 44 features, I used 42 to predict the 44th 'Attack' category. The benign data represented approximately 96% of the data, followed by Exploits. I removed the label category as it represented whether it was benign or not. The distribution of attacks look as follows:


<img width="491" alt="image" src="https://github.com/CSC380-SU23-UofArizona/final-project-chr211/assets/28885019/96ad44c7-cbf9-42ed-b907-9760a55b3817"><img width="179" alt="image" src="https://github.com/CSC380-SU23-UofArizona/final-project-chr211/assets/28885019/1c58910c-7c14-488e-8bbe-17764979a860">




## Data Preprocessing

I began by removing any Nan or invalid data like empty data. I then removed any duplicates present. I encoded the IP addresses to integers for model processing. I convered all numeric data to 64 bit integers. I encoded the attack type strings to numbers between 0 and 9. I used 4 methods of outlier detection:

1. Boxplot
   <img width="415" alt="image" src="https://github.com/CSC380-SU23-UofArizona/final-project-chr211/assets/28885019/92f4fa9e-ab8f-4101-9583-9cfac9307a27">

2. Scatterplot
   <img width="398" alt="image" src="https://github.com/CSC380-SU23-UofArizona/final-project-chr211/assets/28885019/84059b34-667d-4dfd-b904-0b8749dd1e23">

3. Zscore
   <img width="647" alt="image" src="https://github.com/CSC380-SU23-UofArizona/final-project-chr211/assets/28885019/09bf00a8-5211-49f3-beb8-c69b117a5a14">


4. Isolation forest
   <img width="675" alt="image" src="https://github.com/CSC380-SU23-UofArizona/final-project-chr211/assets/28885019/0f4b4661-e2e0-49f4-aa9a-aea2b71b3542">

I chose the Z score method for removing outliers from my data set for comparison 'data_outliers_gone'. I compared this with the data set that had the outliers left in, see results section. I found a Z score of 1 std from the mean provided the best results, but values of 2 and 3 also provided improvement.
## Feature Selection/Engineering
I chose two methods of feature reduction: principle componant analysis and Chi-squared. PCA did not help much for this type of classification problem, but I left it in for reference. The second was a chi-squared reduced feature set. This method reduced the set by 
finding which features were correlated and drop one of them from the set.


<img width="539" alt="image" src="https://github.com/CSC380-SU23-UofArizona/final-project-chr211/assets/28885019/cd3239d2-067d-4871-a247-0555239967fd">
The elbow on the Scree plot displayed the Eigenvalues vs number of features. The optimal number of features can be found by drawing a line where the data approximately becomes linear and the variance is minimized. This is around 8 features in this plot. 


Below is a heatmap that shows the correlation between features in magnitude (might be difficult to read here)

<img width="519" alt="image" src="https://github.com/CSC380-SU23-UofArizona/final-project-chr211/assets/28885019/4abcfc4b-c6aa-4be7-bb9a-9d79dc1adffa">



## Model Selection

I chose 3 models because of their ability to accurately model non-linear classification data. Random forest classifier, Multilayer perceptron classifier neural network, and a support vector machine. The data set has discrete 10 classes of attack data and I desire discrete results. All three performed well in validaton and testing on several variations of the data set. The first model I chose was a random forest classifier. As mentioned earlier, it is a collection of many separate decision trees that each calculate models with associated probabilities and combined with weights. I used 4 different data sets for this model: benign included, a dataset with the 4 most important values removed , no_benign , and PCA_no_benign.

 The second machine learning model I chose was a forward neural network classifier call a multi-layer perceptron classifier. It is a supervised model with a variable number of forwardly interacting layers of nodes that represent the features. The input layer has the same number of nodes as input features and the output layer has the same number of nodes as output classes (attack types in my case). Any hidden layers in between have a variable number of layers. The connections between nodes are assigned weights and activation functions. Forward propagation means that the movement between nodes/neurons is unidirectional between layers. I can choose the number of epochs/iterations for the model.

 The third model was a support vector machine. The SVM works well for classification problems by first dividing the data into hyperplanes that classify the data. Certain groupings of points represent classes. It then finds the best way to place a hyperplane through useful data that fit each type of class so that the difference between the points of a class is maximized.



## Model Training and Evaluation
Each model was trained on the full range of attack data and various subsets. The subsets included benign data removed, normalized full attack data, chi-squared reduced data, and PCA reduced data. The random forest model was trained on a few additional data subsets. The performance between models was compared using the commong subsets.



## Results and Conclusion
The random forest classifier performed slightly better than the forward neural network MLPClassifier on when both were trained on the dataset that included benign data. In fact most of the subsets of the benign-included data performed well with both classifiers. The Benign label data was very useful and important for training to set a baseline of what normal, not malicious, network flow looks like. Models trained on data without benign did not perform as well as those trained with benign network data. For example, the RFC trained on the full data had an accuracy of .98 in one sample compared to an accuracy of .8 with the benign data removed. Both models performed better with normalized data, this help reduce the affect of outlier data reduced the scale of the data for more efficient processing. In section D I compared the data set with no benign data with outliers left in and then with outliers removed using Z scores. The data set with outliers removed improved the accuracy of around 8%. 

Below is the confusion matrix for the RFC trained on all network data, the test accuracy was approximately .98 in most samples.

<img width="689" alt="image" src="https://github.com/CSC380-SU23-UofArizona/final-project-chr211/assets/28885019/4f0a0e82-87db-4478-8d75-d3988c86fb9d">

Below is the confusion matrix for the RFC trained on the set with benign removed. Notice it has more values off the diagonal. The test accuracy was approximately .8 in most samples.


<img width="718" alt="image" src="https://github.com/CSC380-SU23-UofArizona/final-project-chr211/assets/28885019/2a8f4be6-894f-45b3-9084-d0eee138ca39">


Comparatively, the random forest classifier outperformed the mlpclassifier for all common data sets used, (one sample RFC accuracy: .985, while MLPC accuracy was .755 on full data). After researching the differences between the models, RFC are easier to use on non-linear data because of less hyperparameters, compared with MLPC which take more time to determine the optimal values of learning rate and activation functions. MLPC are also more likely to overfit data, although I did not see this between my training and testing results. I will detail how I used each model and which data sets I used below. The final machine learning model I chose was a support vector machine. At present, I only trained it on the full data set. It outperforms the MLPClassifier by 20% accuracy points in some samples. It trails behind the random forest classifier by around 4% accuracy on various samples I used. Therefore, I would choose the random forest classifier for this application with the data sets that had no benign values and outlying values removed with z scores above 1. I would also choose the SVM model. I might be able to improve the performance of the MLPclassifier if I tuned the hyperparameters.

## Limitations
The attack data is limited to attacks on this specific network, both artifical and malicious. There are many more varieties that represent these categories and the models
trained on this data likely would not identify them. I occasionally ran into overflow errors with the MLP classifier when using data sets with outlying values included.


