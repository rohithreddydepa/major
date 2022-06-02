import pickle

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, hamming_loss
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import re

from flask import Flask, render_template, request

global filename
global cls
global title_precision
global title_recall
global title_f1
global cv

stop_words = set(stopwords.words('english'))

tag_names = ['android', 'c#', 'java', 'javascript', 'php']


def getTagID(name):
    tid = -1
    for i in range(len(tag_names)):
        if tag_names[i] == name:
            tid = i
            break
    return tid


def rem_html_tags(question):
    regex = re.compile('<.*?>')
    return re.sub(regex, '', question)


def removePunct(question):
    question = re.sub('\W+', ' ', question)
    question = question.strip()
    return question

app=Flask(__name__)
@app.route("/")
def home():
    return render_template("Userpage.html")

@app.route("/titleModule",methods=["GET","POST"])
def titleModule():
    global cls
    global cv
    global title_precision
    global title_recall
    global title_f1

    train = pd.read_csv('dataset.csv', encoding='iso-8859-1', nrows=50000)
    X = []
    Y = []
    for i in range(len(train)):
        title = train._get_value(i, 'Title')
        title = rem_html_tags(title)
        title = removePunct(title)

        tags = train._get_value(i, 'Tags')
        tags = tags.strip().lower()
        tags = tags.lower()

        data = title
        arr = data.split(" ")
        msg = ''
        for k in range(len(arr)):
            word = arr[k].strip()
            if len(word) > 2 and word not in stop_words:
                msg += word + " "
        texts = msg.strip()
        tag_arr = tags.split(' ')
        class_label = np.zeros(5)
        option = 0
        for k in range(len(tag_arr)):
            tag_id = getTagID(tag_arr[k])
            if tag_id != -1:
                option = 1
                class_label[tag_id] = 1
        if option == 1 and len(X) < 2000:
            Y.append(class_label)
            X.append(texts)

    X = np.asarray(X)
    Y = np.asarray(Y)

    cv = CountVectorizer(analyzer='word', stop_words=stop_words, lowercase=True, ngram_range=(1, 2))
    X = cv.fit_transform(X).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    filename = 'Title_model.sav'
    cls = pickle.load(open(filename, 'rb'))

    y_pred = cls.predict(X_test)

    title_precision = precision_score(y_test, y_pred, average='micro') * 100
    title_recall = recall_score(y_test, y_pred, average='micro') * 100
    title_f1 = f1_score(y_test, y_pred, average='micro') * 100
    acc1 = accuracy_score(y_test, y_pred) * 100
    hl = hamming_loss(y_test, y_pred)
    return render_template("result.html", var=title_precision, var1=title_recall, var2=title_f1, var3=acc1,var4=hl)


@app.route("/bodyModule",methods=["GET","POST"])
def bodyModule():
    global body_precision
    global body_recall
    global body_f1
    train = pd.read_csv('dataset.csv', encoding='iso-8859-1', nrows=50000)
    X = []
    Y = []
    for i in range(len(train)):
        body = train._get_value(i, 'Body')
        body = rem_html_tags(body)
        body = removePunct(body)

        tags = train._get_value(i, 'Tags')
        tags = tags.strip().lower()
        tags = tags.lower()

        data = body
        arr = data.split(" ")
        msg = ''
        for k in range(len(arr)):
            word = arr[k].strip()
            if len(word) > 2 and word not in stop_words:
                msg += word + " "
        texts = msg.strip()

        tag_arr = tags.split(' ')
        class_label = np.zeros(5)
        option = 0
        for k in range(len(tag_arr)):
            tag_id = getTagID(tag_arr[k])
            if tag_id != -1:
                option = 1
                class_label[tag_id] = 1
        if option == 1 and len(X) < 2000:
            Y.append(class_label)
            X.append(texts)

    X = np.asarray(X)
    Y = np.asarray(Y)

    cv = CountVectorizer(analyzer='word', stop_words=stop_words, lowercase=True, ngram_range=(1, 2))
    X = cv.fit_transform(X).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    filename = 'Body_model.sav'
    cls = pickle.load(open(filename, 'rb'))
    y_pred = cls.predict(X_test)

    body_precision = precision_score(y_test, y_pred, average='micro') * 100
    body_recall = recall_score(y_test, y_pred, average='micro') * 100
    body_f1 = f1_score(y_test, y_pred, average='micro') * 100
    acc2 = accuracy_score(y_test, y_pred) * 100
    hl = hamming_loss(y_test, y_pred)
    return render_template("result.html", var=body_precision, var1=body_recall, var2=body_f1, var3=acc2,var4=hl)

@app.route("/bodyAndTitleModule",methods=["GET","POST"])
def bodyAndTitleModule():
    global hand_precision
    global hand_recall
    global hand_f1

    train = pd.read_csv('dataset.csv', encoding='iso-8859-1', nrows=50000)
    X = []
    Y = []
    for i in range(len(train)):
        body = train._get_value(i, 'Body')
        body = rem_html_tags(body)
        body = removePunct(body)

        tags = train._get_value(i, 'Tags')
        tags = tags.strip().lower()
        tags = tags.lower()

        data = body + " " + tags
        arr = data.split(" ")
        msg = ''
        for k in range(len(arr)):
            word = arr[k].strip()
            if len(word) > 2 and word not in stop_words:
                msg += word + " "
        texts = msg.strip()

        tag_arr = tags.split(' ')
        class_label = np.zeros(5)
        option = 0
        for k in range(len(tag_arr)):
            tag_id = getTagID(tag_arr[k])
            if tag_id != -1:
                option = 1
                class_label[tag_id] = 1
        if option == 1 and len(X) < 2000:
            Y.append(class_label)
            X.append(texts)

    X = np.asarray(X)
    Y = np.asarray(Y)

    cv = CountVectorizer(analyzer='word', stop_words=stop_words, lowercase=True, ngram_range=(1, 2))
    X = cv.fit_transform(X).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    filename = 'BodyandTitle_model.sav'
    cls = pickle.load(open(filename, 'rb'))
    y_pred = cls.predict(X_test)

    hand_precision = precision_score(y_test, y_pred, average='micro') * 100
    hand_recall = recall_score(y_test, y_pred, average='micro') * 100
    hand_f1 = f1_score(y_test, y_pred, average='micro') * 100
    acc3 = accuracy_score(y_test, y_pred) * 100
    hl = hamming_loss(y_test, y_pred)
    return render_template("result.html", var=hand_precision, var1=hand_recall, var2=hand_f1, var3=acc3,var4=hl)

@app.route("/recallGraph",methods=["GET","POST"])
def recallGraph():
    height = [title_recall,body_recall,hand_recall]
    bars = ('Title Recall','Body Recall','Hand-Engineered Recall')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Recall graph", loc='center')
    plt.show()
    return render_template("Userpage.html")

@app.route("/FGraph",methods=["GET","POST"])
def FGraph():
    height = [title_f1,body_f1,hand_f1]
    bars = ('Title F1','Body F1','Hand-Engineered F1')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Fgraph", loc='center')
    plt.show()
    return render_template("Userpage.html")

@app.route("/precisionGraph",methods=["GET","POST"])
def precisionGraph():
    height = [title_precision,body_precision,hand_precision]
    bars = ('Title Precision','Body Precision','Hand-Engineered Precision')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Precision", loc='center')
    plt.show()
    return render_template("Userpage.html")

@app.route("/predict",methods=["GET","POST"])
def predict():
            finaltags=[]
            line=request.form["titlebody"]
            print(line);
            line = line.strip('\n')
            line = line.strip()
            temp = line
            line = line.lower()
            if len(line) > 0:
                cv1 = CountVectorizer(vocabulary=cv.get_feature_names(),stop_words = "english", lowercase = True,ngram_range=(1, 2))
                test1 = cv1.fit_transform([line])
                predict = cls.predict(test1.toarray())[0]
                print(predict)
                for i in range(len(predict)):
                    if predict[i] == 1:
                        finaltags.append(tag_names[i])
                print(predict)
                finaltags="Tags identified as: "+finaltags[0];
            return render_template("userpage.html", var=finaltags)

if __name__ == "__main__":
    app.run(port=9000,debug=True)