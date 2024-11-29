import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from pycaret.classification import *
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

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

    train, test = train_test_split(df,test_size=0.3)
    # Set up PyCaret
    clf_setup = setup(
        data=train,
        test_data=test,
        verbose=False,
        target = 'labels',
        n_jobs=1
    )

    #top models printed at the end of comparison
    best_models = compare_models(n_select=5, sort='F1_weighted')
    comparison_results = pull()

    final_model = finalize_model(best_models[0])


    with open(f'{reports_dir}evaluation_report.txt', 'w') as f:
        f.write("Top models and their F1_weighted scores:\n")
        for i, model in enumerate(best_models):
            model_name = type(model).__name__  # Get the model's name
            f1_weighted_score = comparison_results.loc[comparison_results.index[i], 'F1_weighted']
            f.write(f"{i + 1}. {model_name}: {f1_weighted_score:.4f}\n")

    save_model(final_model,f"{models_dir}best_model",)

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
