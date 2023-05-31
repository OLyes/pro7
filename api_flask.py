from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import json


app = Flask(__name__)

pickle_in=open('models/classifier.pkl','rb')
classifier = pickle.load(pickle_in)

# scoring
def score_record(df_all, classifier):

    return classifier.predict(df_all)[0], classifier.predict_proba(df_all)[:,1][0]

@app.route('/')
def hello_world():
    return 'Hello world! How are you?'
    
@app.route('/predict', methods=['GET','POST'])
def predict_score():
    
    #Recuperation des données 
    df_all = request.get_json()
    df_all = np.array(df_all)
    df_all = df_all.reshape(1, -1)

    # utilisation des données pour faire une prediction
    prediction, probability = score_record(df_all, classifier)
    data_df = pd.DataFrame(columns=['prediction','probability'])
    data_df = data_df.append({'prediction' : prediction}, ignore_index=True)
    data_df['probability'] = probability
    output = data_df.to_dict(orient='rows')
    
    return  jsonify(output)


if __name__=='__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
