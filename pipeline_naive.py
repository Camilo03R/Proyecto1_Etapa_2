import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from joblib import dump
from preprocesamiento import Preprocessor  # Importamos nuestro preprocesador personalizado


# Cargar y preparar los datos originales

df = pd.read_csv("fake_news_spanish.csv", sep=";")  # Cargar dataset
df['texto'] = df['Descripcion'] + " " + df['Titulo']  # Combinar descripción y título

# Rellenar valores faltantes y asegurarse de que los textos son strings
X = df['texto'].fillna("").astype(str)
y = df['Label']  # Variable objetivo (0: Falsa, 1: Verdadera)


# Crear el pipeline completo para procesamiento + modelo

pipeline = Pipeline([
    ('preprocessing', Preprocessor()),     # Limpieza y normalización de texto
    ('tfidf', TfidfVectorizer()),          # Vectorización con TF-IDF
    ('clf', MultinomialNB())               # Clasificador Naive Bayes
])


# Entrenar el modelo con el dataset

pipeline.fit(X, y)

# Guardar el pipeline entrenado (para usar en la API)

dump(pipeline, "modelos/naive_bayes.joblib")


