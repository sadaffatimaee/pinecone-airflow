from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from pinecone import Pinecone, ServerlessSpec

def _create_index(**_):
    api = Variable.get("PINECONE_API_KEY")
    name = Variable.get("PINECONE_INDEX", default_var="semantic-search-fast")
    cloud = Variable.get("PINECONE_CLOUD", default_var="aws")
    region = Variable.get("PINECONE_REGION", default_var="us-east-1")

    pc = Pinecone(api_key=api)
    existing = [ix["name"] for ix in pc.list_indexes()]
    if name not in existing:
        print(f"Creating index {name} (cloud={cloud}, region={region})")
        pc.create_index(
            name=name,
            dimension=384,  # for sentence-transformers all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
    else:
        print(f"Index {name} already exists")

    pc.describe_index(name)
    print("Index ready:", name)

with DAG(
    dag_id="create_pinecone_index_dag",
    start_date=datetime(2024,1,1),
    schedule_interval=None,
    catchup=False,
    tags=["pinecone","setup"],
) as dag:
    create_index = PythonOperator(task_id="create_index", python_callable=_create_index)
