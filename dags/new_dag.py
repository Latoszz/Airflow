import base64
import json
import pickle
from datetime import datetime, timedelta

import gspread
import kagglehub
import numpy as np
import pandas as pd
import requests
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from google.oauth2.service_account import Credentials
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

Sheets_api_info = json.loads(Variable.get("Sheets_api_info"))
SHEETS_ID = Variable.get("SHEETS_ID")  #
sheet_name = "data_student_24435"

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive"
]

credentials = Credentials.from_service_account_info(Sheets_api_info, scopes=scope)
client = gspread.authorize(credentials)

def get_dataframe_from_sheet(sheet_from, **kwargs):
    try:
        # get the worksheet
        worksheet = client.open_by_key(SHEETS_ID).worksheet(sheet_from)
        # put data into df
        df = pd.DataFrame(worksheet.get_all_records())
        print(f"Downloaded sheet '{sheet_from}' cells.")

        kwargs['ti'].xcom_push(key=f'{sheet_from}_data', value=df)

    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")


def upload_dataframe_to_sheet(df, SHEET_NAME):
    # Convert the DataFrame to a list of lists (Google Sheets API format)
    data = [df.columns.tolist()] + df.values.tolist()  # Add headers
    try:
        # get the worksheet
        worksheet = client.open_by_key(SHEETS_ID).worksheet(SHEET_NAME)
        # clear it
        x = datetime.now()
        print(f"Cleared sheet '{SHEET_NAME}'   {x.strftime("%X")}")
        worksheet.clear()
        # insert the data
        worksheet.update(data, raw=True)
        elapsed = datetime.now() - x
        dt = timedelta(seconds=elapsed.total_seconds())
        print(f"Updated sheet '{SHEET_NAME}' cells. " + str(dt))
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")


def upload_data_task(sheet_from, sheet_to, **kwargs):
    ti = kwargs['ti']
    df = ti.xcom_pull(key=f'{sheet_from}_data', task_ids=f'prepare_data_{sheet_from}')
    upload_dataframe_to_sheet(df, sheet_to)


def upload_train_data(**kwargs):
    ti = kwargs['ti']
    train_df = ti.xcom_pull(key='train_df', task_ids='split_data')
    upload_dataframe_to_sheet(train_df, SHEET_NAME="Train_raw")


def upload_test_data(**kwargs):
    ti = kwargs['ti']
    test_df = ti.xcom_pull(key='test_df', task_ids='split_data')

    upload_dataframe_to_sheet(test_df, SHEET_NAME="Test_raw")


def download_data(**kwargs):
    # Download dataset
    path = kagglehub.dataset_download("aadyasingh55/sexism-detection-in-english-texts")
    dev_data = pd.read_csv(f'{path}/dev.csv')
    train_data = pd.read_csv(f'{path}/train (2).csv')
    test_data = pd.read_csv(f'{path}/test (1).csv')
    # Combine the data
    df = pd.concat([dev_data, train_data, test_data], ignore_index=True)

    kwargs['ti'].xcom_push(key='dataframe', value=df)


def split_data(**kwargs):
    ti = kwargs['ti']
    df = ti.xcom_pull(key='dataframe', task_ids='download_data')
    train_df, test_df = train_test_split(df, test_size=0.3)
    ti.xcom_push(key='train_df', value=train_df)
    ti.xcom_push(key='test_df', value=test_df)


def drop_na_task(sheet_from, **kwargs):
    ti = kwargs['ti']
    df = ti.xcom_pull(key=f'{sheet_from}_data', task_ids=f'download_{sheet_from}')

    df.replace('', np.nan, inplace=True)
    df.dropna(how='any', inplace=True)

    ti.xcom_push(key=f'{sheet_from}_data', value=df)


def modify_labels_task(sheet_from, **kwargs):
    ti = kwargs['ti']
    df = ti.xcom_pull(key=f'{sheet_from}_data', task_ids=f'drop_na_{sheet_from}')
    df['label_vector'] = df['label_vector'].str.replace(r'^\d+(\.\d+)?\s+', '-', regex=True)

    df["labels"] = df[["label_sexist", "label_category", "label_vector"]].apply(lambda x: ' '.join(x), axis=1)

    df['labels'] = df['labels'].replace('not sexist none none', 'not sexist')
    df = df[["text", "labels"]]

    ti.xcom_push(key=f'{sheet_from}_data', value=df)


# Vectorize data
def prepare_data(sheet_from, vectorizer=None, **kwargs):
    ti = kwargs['ti']
    df = ti.xcom_pull(key=f'{sheet_from}_data', task_ids=f'modify_labels_{sheet_from}')

    # The vectorizer needs to be fitted on the train data, and then used to transform the test data
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 1), stop_words="english", max_features=400
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

    ti.xcom_push(key=f'{sheet_from}_data', value=df)
    ti.xcom_push(key=f'{sheet_from}_vectorizer', value=serialized_vectorizer)


def serialize_vectorizer(vectorizer):
    """Serialize TfidfVectorizer to base64 string."""
    return base64.b64encode(pickle.dumps(vectorizer)).decode('utf-8')


def deserialize_vectorizer(vectorizer_str):
    """Deserialize base64 string back to TfidfVectorizer."""
    return pickle.loads(base64.b64decode(vectorizer_str.encode('utf-8')))


with DAG(
        dag_id="data_download_push_to_sheets",
        start_date=datetime(2024, 11, 24),
        schedule_interval=None,
        catchup=False,
) as dag:
    # Get from kaggle
    download_data_task = PythonOperator(
        task_id='download_data',
        python_callable=download_data,
        provide_context=True,
    )
    # 70, 30 split
    split_data_task = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
        provide_context=True,
    )
    # Push train to sheets
    upload_train_task = PythonOperator(
        task_id='upload_train_data',
        python_callable=upload_train_data,
        provide_context=True,
    )
    # Push test to sheets
    upload_test_task = PythonOperator(
        task_id='upload_test_data',
        python_callable=upload_test_data,
        provide_context=True,
    )

    # Set task dependencies
    download_data_task >> split_data_task >> [upload_train_task, upload_test_task]

with DAG(
        dag_id="data_preparation_pipeline",
        start_date=datetime(2024, 11, 24),
        schedule_interval=None,
        catchup=False,
) as dag:
    # Train dataset tasks
    # Get from sheets
    download_train = PythonOperator(
        task_id='download_Train_raw',
        python_callable=get_dataframe_from_sheet,
        op_kwargs={'sheet_from': 'Train_raw'},
        provide_context=True,
    )
    # drop empty
    drop_na_train = PythonOperator(
        task_id='drop_na_Train_raw',
        python_callable=drop_na_task,
        op_kwargs={'sheet_from': 'Train_raw'},
        provide_context=True,
    )
    # remove excessive numbers and hyphens, Combine the 3 labels into 1 label column, drop columns without data or labels
    modify_labels_train = PythonOperator(
        task_id='modify_labels_Train_raw',
        python_callable=modify_labels_task,
        op_kwargs={'sheet_from': 'Train_raw'},
        provide_context=True,
    )
    # Vectorize data
    prepare_data_train = PythonOperator(
        task_id='prepare_data_Train_raw',
        python_callable=prepare_data,
        op_kwargs={'sheet_from': 'Train_raw'},
        provide_context=True,
    )
    # Upload to sheets
    upload_train = PythonOperator(
        task_id='upload_Train_prepared',
        python_callable=upload_data_task,
        op_kwargs={'sheet_from': 'Train_raw', 'sheet_to': 'Train_prepared'},
        provide_context=True,
    )

    # Test dataset tasks
    # Repeat of all the previous steps
    download_test = PythonOperator(
        task_id='download_Test_raw',
        python_callable=get_dataframe_from_sheet,
        op_kwargs={'sheet_from': 'Test_raw'},
        provide_context=True,
    )
    drop_na_test = PythonOperator(
        task_id='drop_na_Test_raw',
        python_callable=drop_na_task,
        op_kwargs={'sheet_from': 'Test_raw'},
        provide_context=True,
    )
    modify_labels_test = PythonOperator(
        task_id='modify_labels_Test_raw',
        python_callable=modify_labels_task,
        op_kwargs={'sheet_from': 'Test_raw'},
        provide_context=True,
    )
    # We need to pass the vectorizer created in  prepare_data_train
    prepare_data_test = PythonOperator(
        task_id='prepare_data_Test_raw',
        python_callable=prepare_data,
        op_kwargs={'sheet_from': 'Test_raw',
                   'vectorizer': "{{ ti.xcom_pull(task_ids='prepare_data_Train_raw', key='Train_raw_vectorizer') }}"},
        provide_context=True,
    )
    upload_test = PythonOperator(
        task_id='upload_Test_prepared',
        python_callable=upload_data_task,
        op_kwargs={'sheet_from': 'Test_raw', 'sheet_to': 'Test_prepared'},
        provide_context=True,
    )

    # Define task dependencies
    # we need for the prepare_data_train, to finish before starting prepare_data_test, thus the graph cannot be two separate task queues
    download_train >> drop_na_train >> modify_labels_train >> prepare_data_train >> upload_train
    download_test >> drop_na_test >> modify_labels_test
    [prepare_data_train, modify_labels_test] >> prepare_data_test >> upload_test
