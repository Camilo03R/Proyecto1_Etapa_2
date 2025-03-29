import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from joblib import dump
import time
from preprocesamiento import Preprocessor

MODELS = {
    "logistic_regression": LogisticRegression(class_weight={0: 1.2, 1: 0.8}, penalty='l2', solver='sag'),
    "xgboost": XGBClassifier(n_estimators=70, max_depth=7),
    "random_forest": RandomForestClassifier(n_estimators=70, max_depth=None),
    "naive_bayes": MultinomialNB(alpha=0.5)
}

def main():
    start_time = time.time()

    # Cargar datos
    df = pd.read_csv("fake_news_spanish.csv", sep=";")
    df['texto'] = df['Descripcion'].fillna("") + " " + df['Titulo'].fillna("")
    y = df['Label']

    # Crear y entrenar pipelines
    for name, model in MODELS.items():
        print(f"[INFO] Creando pipeline y entrenando modelo {name}...")
        model_time = time.time()

        pipeline = Pipeline([
            ('preprocessing', Preprocessor()),
            ('tfidf', TfidfVectorizer()),
            ('classifier', model)
        ])

        print(f"[INFO] Entrenando modelo {name}...")
        pipeline.fit(df['texto'], y)

        print(f"[INFO] Guardando modelo {name}...")
        dump(pipeline, f"models/{name}_pipeline.joblib")
        print(f"[INFO] Modelo {name} guardado en models/{name}_pipeline.joblib")
        print(f"[INFO] Tiempo de entrenamiento para {name}: {time.time() - model_time:.2f} segundos")

    print(f"[INFO] Tiempo total: {time.time() - start_time:.2f} segundos")
    print("[INFO] Pipelines guardados correctamente.")


if __name__ == '__main__':
    main()