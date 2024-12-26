from nltk.metrics import precision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing WordCloud for text visualiztion
from pandas.core.common import random_state
from wordcloud import WordCloud

#Importing NLTK for natural language processing
import nltk
from nltk.corpus import stopwords # For stopwords
from nltk.tokenize import TreebankWordTokenizer
#Downlaoding NLTK data
nltk.download('stopwords')  #Downloading stopwords data
nltk.download('punkt')      #Downloading tokenizer data

#Read the CSV file
df = pd.read_csv('/home/ankit/ML-Pipeline-/experiment/spam.csv',encoding='latin-1')
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)

#Renmae the columns name
df.rename(columns={'v1':'target','v2':'text'},inplace=True)


## DATA PROCESSING ##
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

#check duplicate values
df.duplicated().sum()

#reomve ducplicate
df = df.drop_duplicates(keep='first')


## FEATURE ENGG ##
#Importing the porter stemmer for text stemming
from nltk.stem.porter import PorterStemmer

#Importing the string module for handlinf special characters
import string

#Creating an insatnce of the Porter Stemmer
ps = PorterStemmer()

#Lowercase transfromation and text preprocesing function
def transform_text(text):
    tokenizer = TreebankWordTokenizer()
    #Transform the text to lowercase
    text = text.lower()

    #Tokenization using NLTK
    text = text = tokenizer.tokenize(text)

    #Removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    #Removing stop words and punctuation
    text = y[:]
    y.clear()

    #Lopp through the tokens and remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    #Stemming using Porter Stemmer
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    #Join the processed tokens back into a single string
    return " ".join(y)

df['transformed_text'] = df['text'].apply(transform_text)

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
tfid = TfidfVectorizer(max_features=500)

X = tfid.fit_transform(df['transformed_text']).toarray()
y = df['target'].values
## TRAIN TEST SPLIT ##
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=2)


## MODEL TRAINING ##
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

svc = SVC(kernel='sigmoid',gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear',penalty='l1')
rfc = RandomForestClassifier(n_estimators=50,random_state=2)
abc = AdaBoostClassifier(n_estimators=50,random_state=2)
bc = BaggingClassifier(n_estimators=50,random_state=2)
etc = ExtraTreesClassifier(n_estimators=50,random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)

clfs = {
    'SVC' : svc,
    'KNN' : knc,
    'NB' : mnb,
    'DT' : dtc,
    'LR' : lrc,
    'RF' : rfc,
    'Adaboost' : abc,
    'Bgc' : bc,
    'ETC' : etc,
    'GBDT' : gbdt,
    'xgb' : xgb
}


## MODEL EVALUATION ##
from sklearn.metrics import accuracy_score,precision_score
def train_classifier(clfs,X_train,X_test,y_train,y_test):
    clfs.fit(X_train,y_train)
    y_pred = clfs.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    return accuracy,precision


accuracy_scores = []
precision_scores = []
for name,clfs in clfs.items():
    current_accuracy,current_precision = train_classifier(clfs,X_train,X_test,y_train,y_test)
    print()
    print("For:",name)
    print("Accuracy:",current_accuracy)
    print("Precision:",current_precision)

    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

if __name__ == '__main__':
    print('X_train shape',X_train.shape)
