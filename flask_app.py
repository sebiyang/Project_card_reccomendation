from flask_restful import reqparse
from flask import Flask, request, jsonify, render_template
from flask import Flask, jsonify

import numpy as np
import pickle as pickle
import json

# Create flask app
app = Flask(__name__)
model = pickle.load(open("multi.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    
    data_list = []
    for data in request.form.values():
        data_list.append(data)

    #data_list[0] = [data_list[0]]
    data_list[0] = int(data_list[0])
    data_list[1] = int(data_list[1])
    data_list[2] = int(data_list[2])
    data_list[3] = int(data_list[3])
    data_list[4] = int(data_list[4])
    data_list[5] = int(data_list[5])

    data_list = [data_list]

    prediction = int(model.predict(data_list))

    labels = ['골든라이프올림카드 ,병원 & 마트 할인으로 당신의 골든라이프를 응원합니다!',
              'KB Pay 챌린지카드,#간편결제 #택시 #편의점 #쇼핑 #배달',
               '탄탄대로 올쇼핑 티타늄카드, 여기저기, 빈틈없이 챙겨받는 올쇼핑 티타늄 할인!',
               '마이핏카드 ,#외식 #편의점 #마트 #주유 #통신 #쇼핑 #배달',
              'Easy study 티타늄카드 ,꿈, 배움, 키움 그리고 이룸 Easy study로 더 쉬움~']

    return render_template("index.html", prediction_text = labels[prediction])


if __name__ == '__main__':
    app.run(debug=True, port=9090)
