import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from pycaret.classification import setup, compare_models, finalize_model, predict_model, save_model, plot_model
from sklearn.metrics import accuracy_score, classification_report

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'start_date': datetime(2024, 1, 1),
    'schedule_interval': None,
    'catchup': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}
home_dir= '/opt/airflow/'

def train_model_with_pycaret():
    df = pd.read_csv(f'{home_dir}processed_data/processed_data.csv')

    unique_labels = df['labels'].unique()
    print(f"Unique labels: {unique_labels}")

    reports_dir = f'{home_dir}reports/'
    models_dir = f'{home_dir}models/'
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Set up PyCaret
    clf_setup = setup(
        data=df,
        train_size=0.7,
        verbose=False,
        target = 'labels',
        n_jobs=1
    )

    # Compare models and get the top 3
    best_models = compare_models(n_select=3, sort='F1_weighted')
    final_model = finalize_model(best_models[0])

    reports_dir = f'{home_dir}reports/'
    os.makedirs(reports_dir, exist_ok=True)

    # Evaluate and log metrics for the top 3 models

if __name__ == '__main__':
    home_dir=''
    train_model_with_pycaret()

with DAG(
        dag_id='model_training_dag',
        default_args=default_args
) as dag:
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model_with_pycaret
    )
