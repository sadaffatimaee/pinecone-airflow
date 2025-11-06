from datetime import datetime
import os, pandas as pd, requests
from airflow import DAG
from airflow.operators.python import PythonOperator

DATA_URL = "https://s3-geospatial.s3.us-west-2.amazonaws.com/medium_data.csv"
LOCAL_DIR = "/opt/airflow/include"
RAW_FILE = os.path.join(LOCAL_DIR, "medium_data.csv")
PREPPED_FILE = os.path.join(LOCAL_DIR, "medium_data_prepped.csv")

def _download(**_):
    os.makedirs(LOCAL_DIR, exist_ok=True)
    r = requests.get(DATA_URL, timeout=60)
    r.raise_for_status()
    with open(RAW_FILE, "wb") as f:
        f.write(r.content)
    print(f"Downloaded to {RAW_FILE}")

def _preprocess(**_):
    df = pd.read_csv(RAW_FILE)
    # ensure required columns exist and an id
    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index":"id"})
    for col in ["title","subtitle"]:
        if col not in df.columns:
            df[col] = ""
    # build the exact columns we’ll use later for Pinecone
    df["metadata"] = df.apply(lambda r: {"title": f"{r['title']} {r['subtitle']}".strip()}, axis=1)
    out = df[["id","title","subtitle","metadata"]].copy()
    os.makedirs(LOCAL_DIR, exist_ok=True)
    out.to_csv(PREPPED_FILE, index=False)
    print(f"Wrote {len(out)} rows to {PREPPED_FILE}")

with DAG(
    dag_id="build_input_file_dag",
    start_date=datetime(2024,1,1),
    schedule_interval=None,
    catchup=False,
    tags=["prep","pinecone"]
) as dag:
    download = PythonOperator(task_id="download_data", python_callable=_download)
    preprocess = PythonOperator(task_id="preprocess_data", python_callable=_preprocess)
    download >> preprocess
