from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import pickle
import os

# Limiter nombre de threads pour eviter le probleme HRAKIRI sur pythonanywhere
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

app = Flask(__name__)

# Chargement du modèle
with open('models/classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Chargement des données
df_all = pd.read_csv('df_all.csv', index_col='SK_ID_CURR')
X = df_all.drop(columns=['TARGET'])
feature_names = X.columns

def score_record(df_all, classifier):
    return classifier.predict(df_all)[0], classifier.predict_proba(df_all)[:,1][0]

@app.route('/')
def home():
    """
    Returns Test home
    """
    return render_template('index.html')

@app.route('/get_value/<int:index>', methods=['GET'])
def get_value(index):
    """
    Returns client informations 
    """
    value = X.loc[index]
    value_dict = value.to_dict()
    return render_template('client.html', value=value_dict, client_id=index)


@app.route('/predict/<int:sk_id>', methods=['GET'])
def predict(sk_id):
    # Vérifier si 'SK_ID_CURR' est l'index du DataFrame
    if 'SK_ID_CURR' not in df_all.index.names:
        return jsonify({"error": "L'index 'SK_ID_CURR' n'existe pas dans le DataFrame."}), 400

    # Filtrer les données pour l'ID spécifié en utilisant .loc
    data = X.loc[[sk_id]]

    # Vérifier si des données ont été trouvées pour l'ID spécifié
    if data.empty:
        return jsonify({"error": "Aucune donnée trouvée pour l'ID spécifié."}), 404

    # Utiliser les fonctions de prédiction pour obtenir le score
    prediction, probability = score_record(data, classifier)

    # Créer un dictionnaire avec les résultats de la prédiction
    result = {
        'client_id': sk_id,
        'probability': round(probability, 4),
        'threshold_value': 0.3918,
        'loan_status': 'Loan denied' if probability > 0.3918 else 'Loan accepted'
    }

    return jsonify(result)

@app.route('/load_classifier', methods=['GET'])
def load_classifier():
    with open('models/classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
        
    classifier_bytes = pickle.dumps(classifier)  # Exporter le modèle en binaire

    return classifier_bytes, 200, {'Content-Type': 'application/octet-stream'}



if __name__ == '__main__':
    app.run()
