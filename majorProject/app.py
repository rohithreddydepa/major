
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
            finaltags=[]
            line=str(request.data,'utf-8')
            line = line.strip('\n')
            line = line.strip()
            cls = TitleBody.cls
            cv = TitleBody.cv
            line = line.lower()
            if len(line) > 0:
                cv1 = CountVectorizer(vocabulary=cv.get_feature_names(),stop_words = "english", lowercase = True,ngram_range=(1, 2))
                test1 = cv1.fit_transform([line])
                predict = cls.predict(test1.toarray())[0]

                for i in range(len(predict)):
                    if predict[i] == 1:
                        finaltags.append(Title.tag_names[i])

                finaltags=finaltags[0];
            return finaltags
@app.route("/graphs",methods=["GET","POST"])
def graphs():
    return jsonify(
        {'Precison':
             {'Title':titleData.get('Precision'),
              'body':bodyData.get('Precision'),
              'TitleBody':titleBodyData.get('Precision')
              },
        'Recall':
            {'Title': titleData.get('Recall'),
             'body': bodyData.get('Recall'),
             'TitleBody': titleBodyData.get('Recall')
             },
         'F1':
             {'Title': titleData.get('F1'),
              'body': bodyData.get('F1'),
              'TitleBody': titleBodyData.get('F1')
              }
         }
    )

if __name__ == "__main__":
    app.run(port=9000, debug=True)
