import base64
import os
import pickle
import re
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from matspy import spy_to_mpl
import nltk
from nltk.tokenize import word_tokenize
from datetime import datetime, timedelta
import kagglehub
import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from sklearn.feature_extraction.text import TfidfVectorizer

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

#[2024-11-26, 19:58:53 UTC] {warnings.py:110} WARNING - /home/***/.local/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1561: UserWarning: Note that pos_label (set to 'sexist 4. prejudiced discussions -supporting systemic discrimination against women as a group') is ignored when average != 'binary' (got 'weighted'). You may use labels=[pos_label] to specify a single positive class.
#TODO why problem with only (this label)
def download_data(**kwargs):
    # Download dataset
    path = kagglehub.dataset_download("aadyasingh55/sexism-detection-in-english-texts")
    dev_data = pd.read_csv(f'{path}/dev.csv')
    train_data = pd.read_csv(f'{path}/train (2).csv')
    test_data = pd.read_csv(f'{path}/test (1).csv')
    # Combine the data
    df = pd.concat([dev_data, train_data, test_data], ignore_index=True)
    print(df.shape)
    kwargs['ti'].xcom_push(key='data', value=df)


def drop_na(**kwargs):
    ti = kwargs['ti']
    df = ti.xcom_pull(key=f'data', task_ids=f'download_data')

    df.replace('', np.nan, inplace=True)
    df.dropna(how='any', inplace=True)
    print(df.shape)

    ti.xcom_push(key=f'data', value=df)

def modify_labels(**kwargs):
    ti = kwargs['ti']
    df = ti.xcom_pull(key=f'data', task_ids=f'drop_na')
    df['label_vector'] = df['label_vector'].str.replace(r'^\d+(\.\d+)?\s+', '-', regex=True)

    df["labels"] = df[["label_sexist", "label_category", "label_vector"]].apply(lambda x: ' '.join(x), axis=1)

    df['labels'] = df['labels'].replace('not sexist none none', 'not sexist')
    df = df[["text", "labels"]]
    print(df.shape)

    ti.xcom_push(key=f'data', value=df)


def drop_dupes(**kwargs):
    ti = kwargs['ti']
    df = ti.xcom_pull(key=f'data', task_ids=f'modify_labels')
    df.drop_duplicates(inplace=True)
    ti.xcom_push(key=f'data', value=df)

def preprocess_text(text):
    text = text.lower()
    text = re.sub("[^a-z0-9]", " ", text)
    text = re.sub("(\s)+", " ", text)
    return text

def custom_tokenizer(text):
    text = preprocess_text(text)
    return word_tokenize(text)

# Vectorize data
def vectorize_data(vectorizer=None, **kwargs):
    nltk.download('punkt_tab')
    ti = kwargs['ti']
    df = ti.xcom_pull(key=f'data', task_ids=f'drop_duplicates')

    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=10,
            max_features=400,
            tokenizer=custom_tokenizer
        )
        X = vectorizer.fit_transform(df['text'])
        serialized_vectorizer = serialize_vectorizer(vectorizer)
    else:
        # Deserialize the vectorizer if it's passed as a string
        if isinstance(vectorizer, str):
            vectorizer = deserialize_vectorizer(vectorizer)
        X = vectorizer.transform(df['text'])
        serialized_vectorizer = serialize_vectorizer(vectorizer)

    print(np.shape(X))

    feature_names = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(X.toarray(), columns=feature_names)

    df = df.drop(columns=['text']).reset_index(drop=True)
    df = pd.concat([df, tfidf_df], axis=1)

    ti.xcom_push(key=f'data', value=df)
    ti.xcom_push(key=f'vectorizer', value=serialized_vectorizer)


def serialize_vectorizer(vectorizer):
    """Serialize TfidfVectorizer to base64 string."""
    return base64.b64encode(pickle.dumps(vectorizer)).decode('utf-8')


def deserialize_vectorizer(vectorizer_str):
    """Deserialize base64 string back to TfidfVectorizer."""
    return pickle.loads(base64.b64decode(vectorizer_str.encode('utf-8')))

def save_to_csv(**kwargs):
    ti = kwargs['ti']
    df = ti.xcom_pull(key=f'data', task_ids=f'vectorize_data')
    processed_data_dir = '/opt/airflow/processed_data/'
    os.makedirs(processed_data_dir, exist_ok=True)

    pd.DataFrame(df).to_csv('/opt/airflow/processed_data/processed_data.csv', index=False)

def visualize(**kwargs):
    ti = kwargs['ti']
    df = ti.xcom_pull(key=f'data', task_ids=f'vectorize_data')
    figure, ax = spy_to_mpl(df.iloc[:, :-1].to_numpy())
    visualizations_dir = '/opt/airflow/visualizations/'
    os.makedirs(visualizations_dir, exist_ok=True)
    figure.savefig('/opt/airflow/visualizations/data.png')

with DAG(
        dag_id="data_processing_dag",
        default_args=default_args
) as dag:
    download_data = PythonOperator(
        task_id='download_data',
        python_callable=download_data,
        provide_context=True,
    )
    drop_na = PythonOperator(
        task_id='drop_na',
        python_callable=drop_na,
        provide_context=True,
    )
    modify_labels = PythonOperator(
        task_id='modify_labels',
        python_callable=modify_labels,
        provide_context=True,
    )
    drop_duplicates = PythonOperator(
        task_id='drop_duplicates',
        python_callable=drop_dupes,
        provide_context=True,
    )
    vectorize_data = PythonOperator(
        task_id='vectorize_data',
        python_callable=vectorize_data,
        provide_context=True,
    )
    save_to_csv = PythonOperator(
        task_id='save_to_csv',
        python_callable=save_to_csv,
        provide_context=True,
    )
    visualize = PythonOperator(
        task_id='visualize',
        python_callable=visualize,
        provide_context=True
    )
    trigger_training = TriggerDagRunOperator(
        task_id='trigger_training',
        trigger_dag_id='model_training_dag',
        wait_for_completion=False
    )
    download_data >> drop_na >> modify_labels >> drop_duplicates >> vectorize_data
    vectorize_data >> [save_to_csv, visualize]
    save_to_csv >> trigger_training

