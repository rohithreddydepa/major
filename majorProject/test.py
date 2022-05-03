import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

stop_words = set(stopwords.words('english'))

tag_names = ['android', 'c#',  'java', 'javascript', 'php']

def getTagID(name):
    tid = -1
    for i in range(len(tag_names)):
        if tag_names[i] == name:
            tid = i
            break;
    return tid    


def rem_html_tags(question):
    regex = re.compile('<.*?>')
    return re.sub(regex, '', question)

def removePunct(question):
    question = re.sub('\W+',' ', question)
    question = question.strip()
    return question


train = pd.read_csv('dataset/dataset.csv',encoding='iso-8859-1',nrows=50000)
print(train.head())


print(list(train.columns.values))

X = []
Y = []

for i in range(len(train)):
    title = train.get_value(i, 'Title')
    title = rem_html_tags(title)
    title = removePunct(title)

    body = train.get_value(i, 'Body')
    body = rem_html_tags(body)
    body = removePunct(body)
    body = body.lower()

    tags = train.get_value(i, 'Tags')
    tags = tags.strip().lower()
    tags = tags.lower()

    data = body+" "+tags
    arr = data.split(" ")
    msg = ''
    for k in range(len(arr)):
        word = arr[k].strip()
        if len(word) > 2 and word not in stop_words:
            msg+=word+" "
    text = msg.strip()

    tag_arr = tags.split(' ')
    class_label = np.zeros(5)#[0,0,0,0,0]
    option = 0
    for k in range(len(tag_arr)):
        tag_id = getTagID(tag_arr[k])
        if tag_id != -1:
            option = 1
            class_label[tag_id] = 1
    if option == 1 and len(X) < 2000:
        Y.append(class_label)
        X.append(text)

X = np.asarray(X)
Y = np.asarray(Y)


'''
count = 0
for i in range(len(Y)):
    if len(Y[i]) > count:
        count = len(Y[i])

temp = np.zeros((len(Y), count))
for i in range(len(Y)):
    for j in range(len(Y[i])):
        temp[i][j] = Y[i][j]

#Y = np.asarray(temp)
Y = to_categorical(temp)
'''
count = 0
for i in range(len(Y)):
    if Y[i][0] == 1:
        count = count + 1
print(count)        

count = 0
for i in range(len(Y)):
    if Y[i][1] == 1:
        count = count + 1
print(count)

count = 0
for i in range(len(Y)):
    if Y[i][2] == 1:
        count = count + 1
print(count)

count = 0
for i in range(len(Y)):
    if Y[i][3] == 1:
        count = count + 1
print(count)

count = 0
for i in range(len(Y)):
    if Y[i][4] == 1:
        count = count + 1
print(count)   
        
print(Y)
print(X.shape)
print(Y.shape)


cv1 = CountVectorizer(analyzer='word',stop_words = stop_words, lowercase = True, ngram_range=(1, 2))
X = cv1.fit_transform(X).toarray()

def prediction(X_test, cls):  #prediction done here
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
        print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
cls = OneVsRestClassifier(LinearSVC())
cls.fit(X_train, y_train)
prediction_data = prediction(X_test, cls) 
y_pred = cls.predict(X_test)

precision = precision_score(y_test, y_pred,average='micro') * 100
recall = recall_score(y_test, y_pred,average='micro') * 100
fmeasure = f1_score(y_test, y_pred,average='micro') * 100


print(precision)
print(recall)
print(fmeasure)

acc = accuracy_score(y_test,y_pred)*100
print(acc)

def processLine(line):
    line = removePunct(line)
    msg = ''
    arr = line.split(' ')
    for i in range(len(arr)):
        arr[i] = arr[i].strip()
        if len(arr[i]) > 2 and arr[i] not in stop_words:
            msg+=arr[i]+" "
    msg = msg.strip();
    print(msg)
    return msg

with open('dataset/test.txt', "r") as file: #reading emotion word
    for line in file:
        line = line.strip('\n')
        line = line.strip()
        temp = line
        line = line.lower()
        line = processLine(line)
        cv = CountVectorizer(vocabulary=cv1.get_feature_names(),stop_words = "english", lowercase = True,ngram_range=(1, 2))
        test1 = cv.fit_transform([line])
        predict = cls.predict(test1.toarray())
        print(predict)










    
    
