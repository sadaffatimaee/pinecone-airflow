from datetime import datetime
import os, json
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

PREPPED_FILE = "/opt/airflow/include/medium_data_prepped.csv"
EMB_PATH = "/opt/airflow/include/embeddings.parquet"

def _make_embeddings(**_):
    if not os.path.exists(PREPPED_FILE):
        raise FileNotFoundError(f"Missing input file: {PREPPED_FILE} (run build_input_file_dag first)")

    df = pd.read_csv(PREPPED_FILE)
    df["text"] = (df["title"].fillna("") + " " + df["subtitle"].fillna("")).str.strip()
    df = df[df["text"].str.len() > 0].copy()
    print(f"Loaded {len(df)} rows to embed")

    model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
    vecs = model.encode(df["text"].tolist(), show_progress_bar=False)

    out = pd.DataFrame({
        "id": df["id"].astype(str),
        "text": df["text"],
        "vector": list(vecs),
        "metadata": df.get("metadata", pd.Series([None]*len(df)))
    })
    out.to_parquet(EMB_PATH, index=False)
    print(f"Wrote embeddings to {EMB_PATH} with shape {out.shape}")

def _upsert_to_pinecone(**_):
    api = Variable.get("PINECONE_API_KEY")
    index_name = Variable.get("PINECONE_INDEX", default_var="semantic-search-fast")
    pc = Pinecone(api_key=api)
    index = pc.Index(index_name)

    df = pd.read_parquet(EMB_PATH)
    print(f"Read {len(df)} vectors for upsert")

    def norm_meta(m):
        if pd.isna(m): return {}
        if isinstance(m, str):
            try: return json.loads(m)
            except Exception: return {"meta": m}
        if isinstance(m, dict): return m
        return {"meta": str(m)}

    batch, B = [], 100
    total = len(df)
    done = 0
    for _, r in df.iterrows():
        batch.append({
            "id": str(r["id"]),
            "values": r["vector"],
            "metadata": {"text": r["text"], **norm_meta(r.get("metadata", {}))}
        })
        if len(batch) == B:
            index.upsert(vectors=batch)
            done += len(batch)
            print(f"Upserted {done}/{total}")
            batch = []
    if batch:
        index.upsert(vectors=batch)
        done += len(batch)
        print(f"Upserted {done}/{total}")

    print(f"Finished upserting {total} vectors into index '{index_name}'")

def _search_example(**_):
    api = Variable.get("PINECONE_API_KEY")
    index_name = Variable.get("PINECONE_INDEX", default_var="semantic-search-fast")
    pc = Pinecone(api_key=api)
    index = pc.Index(index_name)

    query = "data science tips"
    print("Query:", query)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    qvec = model.encode([query])[0].tolist()

    res = index.query(vector=qvec, top_k=5, include_metadata=True)
    print("Top matches:")
    for m in res.get("matches", []):
        txt = (m.get("metadata", {}) or {}).get("text", "")
        print(f"- id={m['id']} score={m['score']:.4f}  text={txt[:120]}")

with DAG(
    dag_id="ingest_and_search_pinecone_dag",
    start_date=datetime(2024,1,1),
    schedule_interval=None,
    catchup=False,
    tags=["pinecone","embeddings","search"],
) as dag:
    make_embeddings = PythonOperator(task_id="make_embeddings", python_callable=_make_embeddings)
    upsert_vectors = PythonOperator(task_id="upsert_vectors", python_callable=_upsert_to_pinecone)
    search_query = PythonOperator(task_id="search_query", python_callable=_search_example)

    make_embeddings >> upsert_vectors >> search_query
