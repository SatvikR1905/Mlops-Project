from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="mlops_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,   # manual trigger
    catchup=False,
) as dag:

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command="cd /opt/mlops && python src/c/preprocess.py"
    )

    train = BashOperator(
        task_id="train",
        bash_command="cd /opt/mlops && python src/c/train.py"
    )

    convert = BashOperator(
        task_id="convert_to_onnx",
        bash_command="cd /opt/mlops && python src/c/convert_to_onmx.py"
    )

    preprocess >> train >> convert
