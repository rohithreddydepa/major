
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
import Title ,TitleBody ,Body

app = Flask(__name__)
titleData=Title.titleModule()
bodyData=Body.bodyModule()
titleBodyData=TitleBody.bodyAndTitleModule()
@app.route("/metrics", methods=["GET", "POST"])
def home():
    args = request.args
    model=args.get('model')
    if model=='Title':
        data = titleData
    elif model=='Body':
        data = bodyData
    elif model=="TitleBody":
        data = titleBodyData
    else:
        data={'error':'Wrong Parameter'}
    return jsonify(data)
@app.route("/predict",methods=["GET","POST"])
def predict():
    finaltags = []
    args = request.args
    line = args.get('data')
    line = line.strip('\n')
    line = line.strip()
    line = line.lower()
    if len(line) > 0:
        cv1 = CountVectorizer(vocabulary=TitleBody.cv.get_feature_names(), stop_words="english", lowercase=True,
                              ngram_range=(1, 2))
        test1 = cv1.fit_transform([line])
        predict = TitleBody.cls.predict(test1.toarray())[0]

        for i in range(len(predict)):
            if predict[i] == 1:
                finaltags.append(Title.tag_names[i])

        finaltags = finaltags[0];
    return jsonify(finaltags)
@app.route("/graphs",methods=["GET","POST"])
def graphs():
    data={}
    data['labels']=['Title','Body','TitleBody'];
    data['datasets'] =[]
    args = request.args
    model = args.get('type')
    tmp = {}
    if model == 'F1':
        tmp['label']='F1 Graph';
        tmp['backgroundColor']='#42A5F5';
        tmp['data']=[titleData.get('values')[2],bodyData.get('values')[2],titleBodyData.get('values')[2]];
    elif model == 'Precision':
        tmp['label']='Precision Graph';
        tmp['backgroundColor']='#FFE0B2';
        tmp['data']=[titleData.get('values')[0],bodyData.get('values')[0],titleBodyData.get('values')[0]];
    elif model == "Recall":
        tmp['label'] = 'Recall Graph';
        tmp['backgroundColor'] = '#42A5F5';
        tmp['data'] = [titleData.get('values')[1], bodyData.get('values')[1], titleBodyData.get('values')[1]];
    data['datasets'].append(tmp);

    return jsonify(data)
if __name__ == "__main__":
    app.run(port=9000, debug=True)
