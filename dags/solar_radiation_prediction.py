from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pickle
import pandas as pd

def load_model():
    # Charger le modèle depuis le fichier .pickle
    with open('/usr/local/airflow/include/model_v2.pickle', 'rb') as file:
        model = pickle.load(file)
    return model

def make_predictions(**kwargs):
    model = load_model()
    
    # Charger les nouvelles données pour faire des prédictions
    new_data = pd.read_csv('/usr/local/airflow/include/df_project_english_v2.csv')
    
    # Utiliser uniquement les colonnes disponibles et nécessaires pour la prédiction (sans 'kWh')
    columns_used_in_model = [
        'surface_pressure', 'snowfall', 'temperature_2m',
        'winddirection_10m', 'relativehumidity_2m', 'windgusts_10m',
        'windspeed_10m', 'precipitation', 'cloudcover', 'elevation'
    ]
    new_data = new_data[columns_used_in_model]
    
    # Faire les prédictions
    predictions = model.predict(new_data)
    
    # Stocker les prédictions dans XCom pour une utilisation ultérieure
    kwargs['ti'].xcom_push(key='predictions', value=predictions.tolist())

def save_predictions(**kwargs):
    # Récupérer les prédictions depuis XCom
    predictions = kwargs['ti'].xcom_pull(key='predictions', task_ids='make_predictions')
    
    # Enregistrer les prédictions dans un fichier CSV
    pd.DataFrame(predictions, columns=['Predictions']).to_csv('/usr/local/airflow/include/predictions.csv', index=False)

# Définition du DAG
with DAG(
    'solar_radiation_prediction',
    default_args={'retries': 1},
    description='Pipeline to load a model and make predictions',
    schedule_interval='@daily',
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    make_predictions_task = PythonOperator(
        task_id='make_predictions',
        python_callable=make_predictions,
        provide_context=True
    )

    save_predictions_task = PythonOperator(
        task_id='save_predictions',
        python_callable=save_predictions,
        provide_context=True
    )

    # Définir les dépendances des tâches
    make_predictions_task >> save_predictions_task
