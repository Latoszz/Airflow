import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from pycaret.classification import setup, compare_models, finalize_model, predict_model, save_model, plot_model
from sklearn.metrics import accuracy_score


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'start_date': datetime(2024, 11, 24),
    'schedule_interval': None,
    'catchup': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def train_model_with_pycaret():
    df = pd.read_csv('/opt/airflow/processed_data/processed_data.csv')

    # Set up PyCaret
    clf_setup = setup(
        data=df,
        target='labels',
        train_size=0.7,
        verbose=False
    )

    # Compare models and get the top 3
    best_models = compare_models(n_select=3)
    final_model = finalize_model(best_models[0])

    reports_dir = '/opt/airflow/reports/'
    os.makedirs(reports_dir, exist_ok=True)

    # Evaluate and log metrics for the top 3 models
    model_accuracies = {}
    for i, model in enumerate(best_models):
        predictions = predict_model(model)
        accuracy = accuracy_score(predictions['Label'], predictions['Score'])
        model_accuracies[f'top_model_{i+1}'] = accuracy

        plot_path = os.path.join(reports_dir, f'top_model_{i+1}_lr.png')
        plot_model(model, plot="lr", save=True)  # Example: Logistic Regression plot
        os.rename("LR.png", plot_path)

    save_model(final_model, '/opt/airflow/models/best_model')

    # Save evaluation metrics and accuracies to a text file
    metrics = predict_model(final_model)[['Label', 'Score']].describe().to_string()
    with open(os.path.join(reports_dir, 'evaluation_report.txt'), 'w') as f:
        f.write("Model Accuracies:\n")
        for model_name, accuracy in model_accuracies.items():
            f.write(f"{model_name}: {accuracy}\n")
        f.write("\nFinal Model Metrics:\n")
        f.write(metrics)

with DAG(
        dag_id='model_training_with_pycaret',
        default_args=default_args
) as dag:
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model_with_pycaret
    )
