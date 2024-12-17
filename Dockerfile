FROM continuumio/miniconda3

RUN pip install mlflow

COPY model /opt/model

CMD ["mlflow", "models", "serve", "-m", "/opt/model", "--no-conda"]
