import numpy as np 
import pandas as pd 

import streamlit as st 
import pickle
import requests
import shap
import json
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from PIL import Image

plt.switch_backend('Agg')

st.set_page_config(
    page_title="Scoring model",
#    page_icon="🎈",
)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.write("""
         (c) 2023
         """)

df = pd.read_csv('df_all.csv',index_col='SK_ID_CURR')
X = df.drop(columns=['TARGET'])
#feature_names = X.columns

# Pickle
with open("models/classifier.pkl","rb") as pickle_in:
    classifier = pickle.load(pickle_in)

#print(X)  # Check


explainer = shap.TreeExplainer(classifier, X, feature_names=X.columns)
#explainer = shap.TreeExplainer(classifier)
shap.initjs()

    
# explain model prediction shap results
def explain_model_prediction_shap(df_all):
    # Calculate Shap values
    shap_values = explainer(np.array(df_all))
    p = shap.plots.bar(shap_values)
    return p, shap_values 

def bivariate_analysis(feat1, feat_2, df_all):
    st.subheader('Bivariate Analysis')
    p = sns.scatterplot(df_all=df_all, x=df_all[feat1], y=df_all[feat_2], hue='TARGET',
                             color='red', s=100)
    return p
 
def plot_gauge(current_value, threshold):
    fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = current_value,
    title = {"text": "Current Value / Threshold Value"},
    gauge = {'axis': {'range': [0, 1]},
             'bar': {'color': "green"},
             'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': threshold}
            }))
    return fig
    
# get the data of the selected customer
def get_value(index):
    # Select the row at the specified index
    value = X.loc[index]
    return value.values.tolist()
  
#def request_prediction(URI, df_all):
    
#    response = requests.post(URI, json=df_all)
#    if response.status_code != 200:
#        raise Exception(
#            "Request failed with status {}, {}".format(response.status_code, response.text))
#   response = json.loads(response.text)
#    response = pd.DataFrame(response)
    
#    prediction = response['prediction'][0]
#    probability = response['probability'][0]
#    result = {'prediction':prediction, 'probability' : probability}
#    return  result

def request_prediction(df_all, classifier):
    df_all = np.array(df_all).reshape(1, -1)
    prediction = classifier.predict(df_all)[0]
    probability = classifier.predict_proba(df_all)[:, 1][0]
    result = {'prediction': prediction, 'probability': probability}
    return result


def best_classification(probas, threshold, X):
    y_pred = 1 if probas > threshold else 0 
    return y_pred

# Définition de la fonction pour afficher les informations du client
def display_client_info(client_data, customer, df):
    st.subheader(f"Data for client {customer}")
    
    # Calcul des statistiques globales pour chaque variable
    statistics = df.describe().transpose()
    
    # Convertir les informations du client en DataFrame
    client_data_df = pd.DataFrame(client_data).transpose()
    
    # Ajouter les informations du client en tant que colonne dans le tableau des statistiques globales
    statistics_with_client = pd.concat([statistics, client_data_df], axis=1)
    
    with st.expander("See data", expanded=False):
        # Affichage des informations du client
        st.write(client_data)
        
        # Affichage des statistiques globales avec les informations du client
        st.subheader("Global Statistics")
        st.write(statistics_with_client)

def display_boxplots(dataframe, selected_id):
    st.header('100 Nearest clients')

    # Créer une instance de NearestNeighbors et ajuster le DataFrame
    nbrs = NearestNeighbors(n_neighbors=101).fit(dataframe.values)

    # Obtenir les indices des 100 clients les plus proches
    distances, indices = nbrs.kneighbors(dataframe.loc[dataframe.index == selected_id].values.reshape(1, -1))

    # Sélectionner les lignes correspondantes pour les 100 clients les plus proches
    top_100_nearest = dataframe.iloc[indices[0][1:]]

    # Afficher les boxplots pour chaque variable et les regrouper par la variable target
    for column in top_100_nearest.columns:
        if column != 'TARGET':
            fig, ax = plt.subplots(figsize=(10, 3))
            sns.boxplot(x=top_100_nearest[column], y=top_100_nearest['TARGET'], ax=ax, showfliers=False, orient='h')
            ax.set_ylabel('TARGET')
            ax.set_title(f'{column}')
            ax.invert_yaxis()
            ax.yaxis.tick_left()
            ax.xaxis.tick_bottom()

            # Ajouter le client sélectionné comme un point sur le boxplot
            selected_customer_value = dataframe.loc[selected_id, column]
            ax.scatter(selected_customer_value, dataframe.loc[selected_id, 'TARGET'], marker='o', color='red', label='Selected Client')

            if column == top_100_nearest.columns[0]:
                ax.legend()

            st.pyplot(fig)


def process():
         
    st.title("Loan Default Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">loan payment risk prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    Customer = st.sidebar.selectbox("Select client number: ",X.index)
    
    if st.sidebar.button("Predict"):
            df_all = get_value(Customer)
            result = request_prediction(df_all, classifier)
            score = result['prediction']
            prob = result['probability']
            y_pred = best_classification(prob, 0.3918, df_all)
            if (y_pred == 1):
                risk_assessment = "Loan denied"
                risk_color = "red"
            else:
                risk_assessment = "Loan accepted"
                risk_color = "green"
            #st.sidebar.success(risk_assessment)
            st.sidebar.markdown(f'<p style="color:{risk_color}">{risk_assessment}</p>', unsafe_allow_html=True)
            st.sidebar.write("Probability: ", round(float(prob),4))
            st.sidebar.write(" best threshold: ", 0.3918)      
            st.subheader('Probability Gauge')
            gauge = plot_gauge(prob, 0.3918)  
            st.plotly_chart(gauge)
            
            st.subheader('Result Interpretability - Applicant Level')
            p, shap_values = explain_model_prediction_shap(df_all) 
            st.pyplot(p)

            st.subheader('Model Interpretability - Overall') 
            #shap_values_ttl = explainer(X) 
            #fig_ttl = shap.plots.bar(shap_values_ttl, max_display=10)
            #st.pyplot(fig_ttl)
            #st.pyplot(shap.summary_plot(shap.TreeExplainer(classifier).shap_values((X)), X, plot_type="bar"))
            shap_image = Image.open(r'globalshap.png')
            st.image(shap_image)
            
            
    if st.sidebar.button('display'):
        # Affichage des boxplots
        display_relative_situation(df)
        
    # Affichage des informations du client sélectionné
    if st.button("Info"):
        # Récupération des informations du client sélectionné
        client_data = df[df.index == Customer]
        # Appel de la fonction pour afficher les informations
        display_client_info(client_data, Customer, df)

    if st.sidebar.checkbox("100 Nearest clients", key=20):
        #st.sidebar.header("100 Nearest clients")
        display_boxplots(df, Customer)

            
def display_relative_situation(df_all):
    st.subheader('Relative Situation - Boxplots')
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.boxplot(data=df_all, ax=ax, showfliers=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)


    
    selected_feature_1 = st.sidebar.selectbox('Feature 1', feature_names)
    selected_feature_2 = st.sidebar.selectbox('Feature 2', feature_names)
    #if st.sidebar.button('display'):
                #data_chart = df.groupby("TARGET")[[selected_feature_1,selected_feature_2]].value_counts().unstack(level=0)
                #st.bar_chart(data_chart)
                #p = bivariate_analysis(selected_feature_1, selected_feature_2, df)
                #st.pyplot()
                

if __name__=='__main__':
    process() 