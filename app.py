import json
from flask import Flask, render_template, request, jsonify
import re
from nltk.corpus import stopwords
import joblib
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

s=set(stopwords.words('english'))
stemmer = SnowballStemmer('english', ignore_stopwords=True)
count=0
classifier = joblib.load('clf.txt')
multibin = joblib.load('multibin.txt')
vectorizer_2=CountVectorizer()

app = Flask(__name__)
data=json.load(open('data.json'))
@app.route('/', methods=['GET'])
def root():
    return render_template('index.html')
@app.errorhandler(404)
def not_found(e):
    return render_template("index.html")
@app.route("/metric", methods=["GET", "POST"])
def metrics():
    args = request.args
    model=args.get('model')
    if(model in data):
        res={'labels':data.get('labels'),'values':data.get(model)}
    else:
        res={'error':'Wrong Parameter'}
    return jsonify(res)
@app.route("/predictTag",methods=["GET","POST"])
def predict():
    question = request.args.get('q')
    print("QUESTION: ",question)
    T=[]
    words = str(question)
    words = re.sub('\n',' ',words)
    words = re.sub('[!@%^&*()$:"?<>=~,;`{}|]',' ',words)
    words = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?]))''',' ',words)
    words = re.sub('_','-',words)
    words = words.replace('[',' ')
    words = words.replace(']',' ')
    words = words.replace('/',' ')
    words = words.replace('\\',' ')
    words = re.sub(r'(\s)\-+(\s)',r'\1', words)
    words = re.sub(r'\.+(\s)',r'\1', words)
    words = re.sub(r'\.+\.(\w)',r'\1', words)
    words = re.sub(r'(\s)\.+(\s)',r'\1', words)
    words = re.sub("'",'', words)
    words = re.sub(r'\s\d+[\.\-\+]+\d+|\s[\.\-\+]+\d+|\s+\d+\s+|\s\d+[\+\-]+',' ',words)
    words = re.sub("^\d+\s|\s\d+\s|\s\d+$"," ", words)
    words = re.sub(r'\s\#+\s|\s\++\s',' ',words)
    stemmed_words = [stemmer.stem(word) for word in words.split()]
    clean_text = filter(lambda w: not w in s,stemmed_words)
    words=''
    for word in clean_text:
            words+=word+' '
    T.append(words)
    print("T",T)
    results=classifier.predict(T)
    results=multibin.inverse_transform(results)
    tagarr=[]
    for result in results[0]:
        tagarr.append(result)
    print(tagarr)
    return (jsonify({"tags":tagarr}))
@app.route("/graph",methods=["GET","POST"])
def graphs():
    args = request.args
    model = args.get('type')
    res={}
    idx=data.get('labels').index(model)
    res['data'] = [data.get('Title')[idx], data.get('Body')[idx], data.get('TitleBody')[idx]];
    return jsonify(res)
@app.route("/preprocessing",methods=["GET","POST"])
def analysis():
    res={}
    res=data.get('count')
    return jsonify(res)

if __name__ == "__main__":
    app.run()
