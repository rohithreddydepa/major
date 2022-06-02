import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import re

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
        # print(texts)
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

    cls = OneVsRestClassifier(LinearSVC())
    cls.fit(X_train, y_train)
    filename = 'Title_model.sav'
    pickle.dump(cls, open(filename, 'wb'))


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

    cls = OneVsRestClassifier(LinearSVC())
    cls.fit(X_train, y_train)
    filename = 'Body_model.sav'
    pickle.dump(cls, open(filename, 'wb'))


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

    cls = OneVsRestClassifier(LinearSVC())
    cls.fit(X_train, y_train)
    filename = 'BodyandTitle_model.sav'
    pickle.dump(cls, open(filename, 'wb'))


titleModule()
bodyModule()
bodyAndTitleModule()
