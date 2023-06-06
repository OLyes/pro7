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

# Définir le backend de Matplotlib pour Streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)
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
feature_names = X.columns

# Pickle
with open("models/classifier.pkl","rb") as pickle_in:
    classifier = pickle.load(pickle_in)


explainer = shap.TreeExplainer(classifier, X, feature_names=feature_names)
shap.initjs()

    
# explain model prediction shap results
def explain_model_prediction_shap(df):
    # Calculate Shap values
    shap_values = explainer(np.array(df))
    p = shap.plots.bar(shap_values)
    return p, shap_values 

def bivariate_analysis(feat1, feat_2, df_all):
    st.subheader('Bivariate Analysis')
    p = sns.scatterplot(data=df_all, x=df_all[feat1], y=df_all[feat_2], hue='TARGET',
                             color='red', s=5)
    return p
 
def plot_gauge(current_value, threshold):
    fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = current_value,
    title = {"text": "Current Value / Threshold Value"},
    gauge = {'axis': {'range': [0, 1]},
             'bar': {'color': "green"},
             'threshold': {'line': {'color': "orange", 'width': 4}, 'thickness': 0.75, 'value': threshold}
            }))
    return fig
    
def get_value(index):
    # Select the row at the specified index
    value = X.loc[index]
    # Convert the row values to a DataFrame with appropriate column names
    value_df = pd.DataFrame([value.values], columns=value.index)
    return value_df

def get_value_shap(index):
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

def display_client_info(client_data, customer, df):

    # Calcul des statistiques globales pour chaque variable en fonction de la valeur de 'TARGET'
    target_value = client_data['TARGET'].values[0]
    if target_value == 0:
        statistics = df[df['TARGET'] == 0].describe().transpose()
    else:
        statistics = df[df['TARGET'] == 1].describe().transpose()

    # Convertir les informations du client en DataFrame
    client_data_df = pd.DataFrame(client_data).transpose()
    client_data_df.columns = client_data_df.columns.astype(str)
    
    # Ajouter les informations du client en tant que colonne dans le tableau des statistiques globales
    statistics_with_client = pd.concat([client_data_df, statistics], axis=1)
    #statistics_with_client.index.astype(str)

    # Affichage des statistiques globales avec les informations du client
    st.subheader(f"Group Stats - TARGET {client_data['TARGET'].values[0]}")
    st.write(statistics_with_client)

    
# Définition de la fonction pour afficher les informations du client
def client_info(client_data):
    st.subheader(f"Client Data")
    
    # Affichage des informations du client
    st.write(client_data)
        

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
    #st.title("Loan Default Prediction")
    html_temp = """
    <div style="background-color:steelblue;padding:10px">
    <h2 style="color:white;text-align:center;">Loan payment risk prediction </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    Customer = st.sidebar.selectbox("Select client number: ", X.index)

    # Récupération des informations du client sélectionné
    client_data = df[df.index == Customer]
    df_all = get_value(Customer)
    df_all_shap = get_value_shap(Customer)

    if st.sidebar.button("Predict"):
        result = request_prediction(df_all, classifier)
        score = result['prediction']
        prob = result['probability']
        y_pred = best_classification(prob, 0.3918, df_all)
        if y_pred == 1:
            risk_assessment = "Loan denied"
            risk_color = "red"
        else:
            risk_assessment = "Loan accepted"
            risk_color = "green"

        st.sidebar.markdown(f'<p style="color:{risk_color}">{risk_assessment}</p>', unsafe_allow_html=True)
        st.sidebar.write("Updated Probability: ", round(float(prob), 4))
        st.sidebar.write("Best threshold: ", 0.3918)
        gauge = plot_gauge(prob, 0.3918)
        gauge.update_traces(gauge=dict(bar=dict(color=risk_color)))
        st.plotly_chart(gauge)



    if st.sidebar.checkbox("Client Data"):
        # Récupération des informations du client sélectionné
        client_data = df[df.index == Customer]
        client_info(client_data)

    # Affichage des informations du client sélectionné
    if st.sidebar.checkbox("Group Stats"):
        # Récupération des informations du client sélectionné
        client_data = df[df.index == Customer]
        # Appel de la fonction pour afficher les informations
        display_client_info(client_data, Customer, df)

        # st.sidebar.header("100 Nearest clients")
        display_boxplots(df, Customer)

    if st.sidebar.checkbox("Feature Importance"):
        st.subheader('Result Interpretability - Applicant Level')
        df_all_shap = get_value_shap(Customer)
        p, shap_values = explain_model_prediction_shap(df_all_shap)
        st.pyplot(p)

        st.subheader('Model Interpretability - Overall')
        # shap_values_ttl = explainer(X)
        # fig_ttl = shap.plots.bar(shap_values_ttl, max_display=10)
        # st.pyplot(fig_ttl)
        # st.pyplot(shap.summary_plot(shap.TreeExplainer(classifier).shap_values((X)), X, plot_type="bar"))
        shap_image = Image.open(r'globalshap.png')
        st.image(shap_image)


    if st.sidebar.checkbox("Edit Client Data"):
        st.subheader("Edit Client Data / Updated Probability")

        # Ajouter les curseurs pour chaque variable que vous souhaitez modifier
        
        EXT_SOURCE_3 = st.sidebar.slider("EXT_SOURCE_3", min_value=0.0, max_value=0.9, value=float(client_data['EXT_SOURCE_3'].values[0]))
        EXT_SOURCE_2 = st.sidebar.slider("EXT_SOURCE_2", min_value=0.0, max_value=1.0, value=float(client_data['EXT_SOURCE_2'].values[0]))
        PAYMENT_RATE = st.sidebar.slider("PAYMENT_RATE", min_value=0.02, max_value=0.13, value=float(client_data['PAYMENT_RATE'].values[0]), step=0.005)
        DAYS_EMPLOYED = st.sidebar.slider("DAYS_EMPLOYED", min_value=-2000.0, max_value=0.0, value=float(client_data['DAYS_EMPLOYED'].values[0]))
        AMT_ANNUITY = st.sidebar.slider("AMT_ANNUITY", min_value=20000.0, max_value=30000.0, value=float(client_data['AMT_ANNUITY'].values[0]))


        # Mettre à jour les valeurs du client sélectionné avec les nouvelles valeurs
        df_all['PAYMENT_RATE'] = PAYMENT_RATE
        df_all['EXT_SOURCE_3'] = EXT_SOURCE_3
        df_all['EXT_SOURCE_2'] = EXT_SOURCE_2
        df_all['DAYS_EMPLOYED'] = DAYS_EMPLOYED
        df_all['AMT_ANNUITY'] = AMT_ANNUITY
        
        # Mettre à jour d'autres variables en fonction des curseurs

        result = request_prediction(df_all, classifier)
        score = result['prediction']
        prob = result['probability']
        y_pred = best_classification(prob, 0.3918, df_all)
        if y_pred == 1:
            risk_assessment = "Loan denied"
            risk_color = "red"
        else:
            risk_assessment = "Loan accepted"
            risk_color = "green"

        st.sidebar.markdown(f'<p style="color:{risk_color}">{risk_assessment}</p>', unsafe_allow_html=True)
        st.sidebar.write("Updated Probability: ", round(float(prob), 4))
        st.sidebar.write("Best threshold: ", 0.3918)
        gauge = plot_gauge(prob, 0.3918)
        gauge.update_traces(gauge=dict(bar=dict(color=risk_color)))
        st.plotly_chart(gauge)



    if st.sidebar.checkbox("Bivariate Analysis"):
         
        selected_feature_1 = st.sidebar.selectbox('Select X', feature_names)
        selected_feature_2 = st.sidebar.selectbox('Select Y', feature_names)
        if st.sidebar.button('display'):
                    #data_chart = df.groupby("TARGET")[[selected_feature_1,selected_feature_2]].value_counts().unstack(level=0)
                    #st.bar_chart(data_chart)
                    df_sample = df.sample(n=5000, random_state=42)
                    p = bivariate_analysis(selected_feature_1, selected_feature_2, df_sample)
                    plt.legend(loc="upper right")
                    st.pyplot()
                
                

if __name__=='__main__':
    process() 