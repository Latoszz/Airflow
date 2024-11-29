import base64
import os
import pickle
import re
from datetime import datetime, timedelta

import kagglehub
import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
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
home_dir= '/opt/airflow/'

def download_data(**kwargs):
    # Download dataset
    path = kagglehub.dataset_download("aadyasingh55/sexism-detection-in-english-texts")
    dev_data = pd.read_csv(f'{path}/dev.csv')
    train_data = pd.read_csv(f'{path}/train (2).csv')
    test_data = pd.read_csv(f'{path}/test (1).csv')
    # Combine the data
    df = pd.concat([dev_data, train_data, test_data], ignore_index=True)
    print(df.shape)
    kwargs['ti'].xcom_push(key=f'data', value=df)
    return df


def drop_na(df=None, **kwargs):
    if df is None:
        ti = kwargs['ti']
        df = ti.xcom_pull(key=f'data', task_ids=f'download_data')

    df= pd.DataFrame(df)
    print(df.shape)

    df.replace('', np.nan, inplace=True)
    df.dropna(how='any', inplace=True)
    print(df.shape)
    kwargs['ti'].xcom_push(key=f'data', value=df)

    return df


def modify_labels(df=None, **kwargs):
    if df is None:
        ti = kwargs['ti']
        df = ti.xcom_pull(key=f'data', task_ids=f'drop_na')
    df= pd.DataFrame(df)
    df['label_vector'] = df['label_vector'].str.replace(r'^\d+(\.\d+)?\s+', '-', regex=True)
    df["labels"] = df[["label_sexist", "label_category", "label_vector"]].apply(lambda x: ' '.join(x), axis=1)

    df['labels'] = df['labels'].replace('not sexist none none', 'not sexist')

    df = df[["text", "labels"]]
    print(df.shape)
    kwargs['ti'].xcom_push(key=f'data', value=df)

    return df


def drop_dupes(df=None, **kwargs):
    if df is None:
        ti = kwargs['ti']
        df = ti.xcom_pull(key=f'data', task_ids=f'modify_labels')
    df = pd.DataFrame(df)

    df.drop_duplicates(inplace=True)

    kwargs['ti'].xcom_push(key=f'data', value=df)

    return df

def preprocess_text(text):
    text = text.lower()
    # Remove ' when not part of a contraction
    text = re.sub(r"(?<![a-zA-Z])'|'(?![a-zA-Z])", " ", text)
    # Replace non-alphanumeric chars (except apostrophes)
    text = re.sub(r"[^a-zA-Z0-9']+", " ",text)
    text = re.sub("(\s)+", " ", text)
    return text

def custom_tokenizer(text):
    text = preprocess_text(text)
    tokens = text.split()
    # Remove single-letter words
    filtered_tokens = [token for token in tokens if len(token) > 1]
    return filtered_tokens

# Vectorize data
def vectorize_data(given_vectorizer=None, df=None, **kwargs):
    if df is None:
        ti = kwargs['ti']
        df = ti.xcom_pull(key=f'data', task_ids=f'drop_duplicates')
    df= pd.DataFrame(df)

    if given_vectorizer is None:
        given_vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=10,
            max_features=100,
            tokenizer=custom_tokenizer
        )
        X = given_vectorizer.fit_transform(df['text'])
        print(np.shape(X))
        vectorizer = serialize_vectorizer(given_vectorizer)
    else:
        # Deserialize the vectorizer if it's passed as a string
        if isinstance(given_vectorizer, str):
            given_vectorizer = deserialize_vectorizer(given_vectorizer)
        X = given_vectorizer.transform(df['text'])
        vectorizer = serialize_vectorizer(given_vectorizer)

    print(np.shape(X))

    feature_names = given_vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(X.toarray(), columns=feature_names)

    df = df.drop(columns=['text']).reset_index(drop=True)
    df = pd.concat([df, tfidf_df], axis=1)
    kwargs['ti'].xcom_push(key=f'data', value=df)
    kwargs['ti'].xcom_push(key=f'vectorizer', value=vectorizer)
    return df,vectorizer

def serialize_vectorizer(vectorizer):
    """Serialize TfidfVectorizer to base64 string."""
    return base64.b64encode(pickle.dumps(vectorizer)).decode('utf-8')


def deserialize_vectorizer(vectorizer_str):
    """Deserialize base64 string back to TfidfVectorizer."""
    return pickle.loads(base64.b64decode(vectorizer_str.encode('utf-8')))

def save_to_csv(df=None, **kwargs):
    if df is None:
        ti = kwargs['ti']
        df = ti.xcom_pull(key=f'data', task_ids=f'vectorize_data')
    df= pd.DataFrame(df)

    processed_data_dir = f'{home_dir}processed_data/'
    os.makedirs(processed_data_dir, exist_ok=True)

    pd.DataFrame(df).to_csv(f'{home_dir}processed_data/processed_data.csv', index=False)



if __name__ == '__main__':
    home_dir = ''
    mydf = download_data()
    mydf = drop_na(mydf)
    mydf = modify_labels(mydf)
    mydf = drop_dupes(mydf)
    mydf = vectorize_data(df=mydf)[0]
    save_to_csv(mydf)

with DAG(
        dag_id="data_processing_dag",
        default_args=default_args
) as dag:
    download_data = PythonOperator(
        task_id='download_data',
        python_callable=download_data
    )
    drop_na = PythonOperator(
        task_id='drop_na',
        python_callable=drop_na
    )
    modify_labels = PythonOperator(
        task_id='modify_labels',
        python_callable=modify_labels
    )
    drop_duplicates = PythonOperator(
        task_id='drop_duplicates',
        python_callable=drop_dupes
    )
    vectorize_data = PythonOperator(
        task_id='vectorize_data',
        python_callable=vectorize_data
    )
    save_to_csv = PythonOperator(
        task_id='save_to_csv',
        python_callable=save_to_csv
    )
    trigger_training = TriggerDagRunOperator(
        task_id='trigger_training',
        trigger_dag_id='model_training_dag',
        wait_for_completion=False
    )
    download_data >> drop_na >> modify_labels >> drop_duplicates >> vectorize_data >> save_to_csv >> trigger_training



