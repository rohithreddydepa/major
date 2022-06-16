import json
from flask import Flask, render_template, request, jsonify

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
    # finaltags = []
    # args = request.args
    # line = args.get('data')
    # line = line.strip('\n')
    # line = line.strip()
    # line = line.lower()
    # if len(line) > 0:
    #     cv1 = CountVectorizer(vocabulary=TitleBody.cv.get_feature_names(), stop_words="english", lowercase=True,
    #                           ngram_range=(1, 2))
    #     test1 = cv1.fit_transform([line])
    #     predict = TitleBody.cls.predict(test1.toarray())[0]

    #     for i in range(len(predict)):
    #         if predict[i] == 1:
    #             finaltags.append(Title.tag_names[i])

    #     finaltags = finaltags[0];
    # return jsonify(finaltags)
    return jsonify('predict is under develpment')
@app.route("/graph",methods=["GET","POST"])
def graphs():
    args = request.args
    model = args.get('type')
    final={}
    res={}
    final['labels']=['Title','Body','TitleBody']
    final['datasets']=[]
    res['label']=model+" Graph"
    res['backgroundColor']='#FFE0B2' if(model=='Precision') else '#42A5F5'
    idx=data.get('labels').index(model)
    res['data'] = [data.get('Title')[idx], data.get('Body')[idx], data.get('TitleBody')[idx]];
    final['datasets'].append(res);
    return jsonify(final)
if __name__ == "__main__":
    app.run()
