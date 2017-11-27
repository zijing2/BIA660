#!flask/bin/python
from flask import Flask, jsonify, request, Response
from ReviewAnalyser import ReviewAnalyser
from flask import render_template
import pandas as pd
from nltk import tokenize

app = Flask(__name__)

@app.route('/', methods=['GET'])
def test_api():
    resp = jsonify({'result': "success"})
    return resp

@app.route('/reviewAnalyser/api/v1.0/predict/label', methods=['POST'])
def predict_label():
    data=pd.read_csv("data_sample.csv",header=None)
    ra = ReviewAnalyser(data)
    ra.pretrain()
    ra.trainLebels(RETRAIN=0)
    label_predict = ra.predictLabels(text_arr=request.json.get("text_arr"))
    return jsonify({'code': 100000, 'data': label_predict}), 201

@app.route('/reviewAnalyser/api/v1.0/predict/sentiment', methods=['POST'])
def predict_sentiment():
    data=pd.read_csv("data_sample.csv",header=None)
    ra = ReviewAnalyser(data)
    ra.pretrain()
    ra.trainSentiment(RETRAIN=0)
    sentiment_predict = ra.predictSentiment(text_arr=request.json.get("text_arr"))
    return jsonify({'code': 100000, 'data': sentiment_predict}), 201

@app.route('/reviewAnalyser/api/v1.0/predict/review', methods=['POST'])
def predict_review():
    reviews = request.json.get("reviews")
    text_arr = []
    for rev in reviews[0:10]:
        rev_sent = tokenize.sent_tokenize(rev)
        for sent in rev_sent:
            text_arr.append(sent)

    data=pd.read_csv("data_sample.csv",header=None)
    ra = ReviewAnalyser(data)
    ra.pretrain()
    ra.trainLebels(RETRAIN=0)
    label_predict = ra.predictLabels(text_arr)
    ra.trainSentiment(RETRAIN=0)
    sentiment_predict = ra.predictSentiment(text_arr)
    resp = jsonify({'code': 100000, 'labels' : label_predict,'sent': sentiment_predict})
    return resp

@app.route('/reviewAnalyser/api/v1.0/performace/label', methods=['GET'])
def performance_labels():
    data=pd.read_csv("data_sample.csv",header=None)
    ra = ReviewAnalyser(data)
    ra.pretrain()
    ra.trainLebels(RETRAIN=0)
    rtn = ReviewAnalyser.checkPerform(ra.label_model, ra.label_mlb, ra.label_padding_sequence, ra.label_act)
    #return jsonify({'result': rtn})
    return rtn

@app.route('/reviewAnalyser/api/v1.0/performace/sent', methods=['GET'])
def performance_sent():
    data=pd.read_csv("data_sample.csv",header=None)
    ra = ReviewAnalyser(data)
    ra.pretrain()
    ra.trainSentiment(RETRAIN=0)
    rtn = ReviewAnalyser.checkPerform(ra.sent_model, ra.sent_mlb, ra.sent_padding_sequence, ra.sent_act)
    #return jsonify({'result': rtn})
    return rtn

@app.route('/reviewAnalyser/api/v1.0/documentInform/1', methods=['GET'])
def performance_img1():
    image = file("1.jpg")
    resp = Response(image, mimetype="image/jpeg")
    return resp

@app.route('/reviewAnalyser/api/v1.0/documentInform/2', methods=['GET'])
def performance_img2():
    image = file("2.jpg")
    resp = Response(image, mimetype="image/jpeg")
    return resp

if __name__ == '__main__':
    app.run(debug=True, port=8887)

