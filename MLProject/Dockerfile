FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir mlflow pandas scikit-learn joblib matplotlib seaborn

CMD ["python", "modelling.py"]