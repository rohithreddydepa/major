
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import simpledialog
from tkinter import filedialog
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import re

main = tkinter.Tk()
main.title('#ML #NLP: Autonomous Tagging of Stack Overflow Question') #designing main screen
main.geometry("1300x1200")

#ML #NLP: Autonomous Tagging of Stack Overflow Question

global filename
global cls
global title_precision,body_precision,hand_precision
global title_recall,body_recall,hand_recall
global title_f1,body_f1,hand_f1
global cv

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



def upload():
    text.delete('1.0', END)
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");


def titleModule():
    global cls
    global cv
    global title_precision
    global title_recall
    global title_f1
    text.delete('1.0', END)
    train = pd.read_csv(filename,encoding='iso-8859-1',nrows=50000)
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
                msg+=word+" "
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
    text.insert(END,'Total questions found in dataset is : '+str(len(X))+"\n")

    cv = CountVectorizer(analyzer='word',stop_words = stop_words, lowercase = True, ngram_range=(1, 2))
    X = cv.fit_transform(X).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    text.insert(END,'Total train size : '+str(X_train.shape)+"\n")
    text.insert(END,'Total test size  : '+str(X_test.shape)+"\n\n")

    cls = OneVsRestClassifier(LinearSVC())
    cls.fit(X_train, y_train)
    #prediction_data = prediction(X_test, cls) 
    y_pred = cls.predict(X_test)

    title_precision = precision_score(y_test, y_pred,average='micro') * 100
    title_recall = recall_score(y_test, y_pred,average='micro') * 100
    title_f1 = f1_score(y_test, y_pred,average='micro') * 100
    acc = accuracy_score(y_test,y_pred)*100

    text.insert(END,'Title Ngrams Precision : '+str(title_precision)+"\n")
    text.insert(END,'Title Ngrams Recall    : '+str(title_recall)+"\n")
    text.insert(END,'Title Ngrams F1        : '+str(title_f1)+"\n")
    text.insert(END,'Title Ngrams Accuracy  : '+str(acc)+"\n")



def body():
    global cls
    global cv
    global body_precision
    global body_recall
    global body_f1
    text.delete('1.0', END)
    train = pd.read_csv(filename,encoding='iso-8859-1',nrows=50000)
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
                msg+=word+" "
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
    text.insert(END,'Total questions found in dataset is : '+str(len(X))+"\n")

    cv = CountVectorizer(analyzer='word',stop_words = stop_words, lowercase = True, ngram_range=(1, 2))
    X = cv.fit_transform(X).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    text.insert(END,'Total train size : '+str(X_train.shape)+"\n")
    text.insert(END,'Total test size  : '+str(X_test.shape)+"\n\n")

    cls = OneVsRestClassifier(LinearSVC())
    cls.fit(X_train, y_train)
    #prediction_data = prediction(X_test, cls) 
    y_pred = cls.predict(X_test)

    body_precision = precision_score(y_test, y_pred,average='micro') * 100
    body_recall = recall_score(y_test, y_pred,average='micro') * 100
    body_f1 = f1_score(y_test, y_pred,average='micro') * 100
    acc = accuracy_score(y_test,y_pred)*100

    text.insert(END,'Body Ngrams Precision : '+str(body_precision)+"\n")
    text.insert(END,'Body Ngrams Recall    : '+str(body_recall)+"\n")
    text.insert(END,'Body Ngrams F1        : '+str(body_f1)+"\n")
    text.insert(END,'Body Ngrams Accuracy  : '+str(acc)+"\n")
                
    
def handengineer():
    global cls
    global cv
    global hand_precision
    global hand_recall
    global hand_f1
    text.delete('1.0', END)
    train = pd.read_csv(filename,encoding='iso-8859-1',nrows=50000)
    X = []
    Y = []
    for i in range(len(train)):
        body = train._get_value(i, 'Body')
        body = rem_html_tags(body)
        body = removePunct(body)

        tags = train._get_value(i, 'Tags')
        tags = tags.strip().lower()
        tags = tags.lower()

        data = body+" "+tags
        arr = data.split(" ")
        msg = ''
        for k in range(len(arr)):
            word = arr[k].strip()
            if len(word) > 2 and word not in stop_words:
                msg+=word+" "
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
    text.insert(END,'Total questions found in dataset is : '+str(len(X))+"\n")

    cv = CountVectorizer(analyzer='word',stop_words = stop_words, lowercase = True, ngram_range=(1, 2))
    X = cv.fit_transform(X).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    text.insert(END,'Total train size : '+str(X_train.shape)+"\n")
    text.insert(END,'Total test size  : '+str(X_test.shape)+"\n\n")

    cls = OneVsRestClassifier(LinearSVC())
    cls.fit(X_train, y_train)
    #prediction_data = prediction(X_test, cls) 
    y_pred = cls.predict(X_test)

    hand_precision = precision_score(y_test, y_pred,average='micro') * 100
    hand_recall = recall_score(y_test, y_pred,average='micro') * 100
    hand_f1 = f1_score(y_test, y_pred,average='micro') * 100
    acc = accuracy_score(y_test,y_pred)*100

    text.insert(END,'Hand Engineered Precision : '+str(hand_precision)+"\n")
    text.insert(END,'Hand Engineered Recall    : '+str(hand_recall)+"\n")
    text.insert(END,'Hand Engineered F1        : '+str(hand_f1)+"\n")
    text.insert(END,'Hand Engineered Accuracy  : '+str(acc)+"\n")

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

def predict():
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="dataset")
    with open(filename, "r") as file: #reading emotion word
        for line in file:
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
                        text.insert(END,temp+" TAG IDENTIFIED AS : "+tag_names[i]+"\n\n")
                print(predict)
    
                

def recallGraph():
    height = [title_recall,body_recall,hand_recall]
    bars = ('Title Recall','Body Recall','Hand-Engineered Recall')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def FGraph():
    height = [title_f1,body_f1,hand_f1]
    bars = ('Title F1','Body F1','Hand-Engineered F1')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

    
    
def precisionGraph():
    height = [title_precision,body_precision,hand_precision]
    bars = ('Title Precision','Body Precision','Hand-Engineered Precision')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()
    

font = ('times', 16, 'bold')
title = Label(main, text='#ML #NLP: Autonomous Tagging of Stack Overflow Question')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=550,y=100)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

titleButton = Button(main, text="Run Title Ngrams, TF-IDF with OneRest & LinearSVM Classifier", command=titleModule)
titleButton.place(x=50,y=150)
titleButton.config(font=font1) 

bodyButton = Button(main, text="Run Body Ngrams, TF-IDF with OneRest & LinearSVM Classifier", command=body)
bodyButton.place(x=50,y=200)
bodyButton.config(font=font1) 

handButton = Button(main, text="Run Hand-engineered, TF-IDF with OneRest & LinearSVM Classifier", command=handengineer)
handButton.place(x=50,y=250)
handButton.config(font=font1) 

precisionButton = Button(main, text="Precision Graph", command=precisionGraph)
precisionButton.place(x=50,y=300)
precisionButton.config(font=font1)

recallButton = Button(main, text="Recall Graph", command=recallGraph)
recallButton.place(x=50,y=350)
recallButton.config(font=font1)

FButton = Button(main, text="F1 Graph", command=FGraph)
FButton.place(x=50,y=400)
FButton.config(font=font1)

predictButton = Button(main, text="Predict Tag from Question", command=predict)
predictButton.place(x=50,y=450)
predictButton.config(font=font1)

#main.config(bg='OliveDrab2')
main.mainloop()
