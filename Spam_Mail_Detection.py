
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# loading the dataset to a Pandas DataFrame
mails =pd.read_csv("mail_data.csv")
# first 5 rows of the dataset
print(mails.head())
# checking for missing values
print(mails.isnull().sum())
mails.where((pd.notnull(mails)),"")

#Labeling ham as 0 and spam as 1
mails['Category']=mails['Category'].map({'ham': 0, 'spam': 1})
# separate the data and Label
X = mails['Message']
Y = mails['Category']
#Splitting the data into train and test data 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
print(Y.shape, Y_train.shape, Y_test.shape)
#transforming the text data into the feature vectors which is used as input 
feature_extraction= TfidfVectorizer(min_df=1,stop_words='english',lowercase='true')
X_train_feature = feature_extraction.fit_transform(X_train)
X_test_feature= feature_extraction.transform(X_test)
# Converting Ytest and train data into integers
Y_train=Y_train.astype("int")
Y_test=Y_test.astype("int")

#Train the Model
LR = LogisticRegression()
LR.fit(X_train_feature, Y_train)
# prediction on train data
X_train_prediction = LR.predict(X_train_feature)
train_data_accuracy = accuracy_score(X_train_prediction, Y_train)*100

print('Logisitic Reg Accuracy : ', train_data_accuracy)

# prediction on test data
X_test_prediction = LR.predict(X_test_feature)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)*100

print('Logisitic Reg Accuracy : ', test_data_accuracy)

#print(LR.coef_.tolist())


input_mail = ["URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!"]
transfo= feature_extraction.transform(input_mail)
predictio = LR.predict(transfo)
print(predictio)
if predictio[0]==0:
    print("ham mail")
else:
    print("spam mail")
