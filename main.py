from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Estructuras de datos para validar las entradas de los endpoints
from structure import DataModel, ReentrenamientoModel

# Para guardar/cargar modelos
from joblib import load, dump

# Métricas de evaluación
from sklearn.metrics import precision_score, recall_score, f1_score

# Componentes del pipeline de ML
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Preprocesador personalizado
from preprocesamiento import Preprocessor

# Utilidades
import os
import pandas as pd

# Inicialización de la aplicación FastAPI
app = FastAPI(title="Clasificación de noticias falsas")

# Permitir acceso desde cualquier frontend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Diccionario de modelos disponibles
MODELOS = {
    "naive_bayes": "models/naive_bayes_pipeline.joblib",
    "logistic_regression": "models/logistic_regression_pipeline.joblib",
    "random_forest": "models/random_forest_pipeline.joblib",
    "xgboost": "models/xgboost_pipeline.joblib",
}

@app.get("/")
def root():
    return {
        "mensaje": "API disponible",
        "modelos_disponibles": list(MODELOS.keys())
    }

@app.post("/predict")
def predecir(data: DataModel):
    modelo_id = data.modelo

    if modelo_id not in MODELOS:
        raise HTTPException(status_code=400, detail="Modelo no soportado")

    ruta = MODELOS[modelo_id]
    if not os.path.exists(ruta):
        raise HTTPException(status_code=404, detail="Modelo no encontrado en el servidor")

    modelo = load(ruta)

    textos = [desc + " " + tit for desc, tit in zip(data.Descripcion, data.Titulo)]

    pred = modelo.predict(textos).tolist()
    prob = modelo.predict_proba(textos).tolist()

    return {
        "modelo_usado": modelo_id,
        "predicciones": pred,
        "probabilidades": prob
    }

@app.post("/retrain")
def reentrenar(data: ReentrenamientoModel):
    modelo_id = data.modelo

    if modelo_id not in MODELOS:
        raise HTTPException(status_code=400, detail="Modelo no soportado")

    ruta = MODELOS[modelo_id]

    try:
        textos = [desc + " " + tit for desc, tit in zip(data.Descripcion, data.Titulo)]
        df = pd.DataFrame({'texto': textos, 'Label': data.Label})

        X = df['texto']
        y = df['Label']

        modelo = load(ruta)

        if modelo_id == "naive_bayes":
            clases = [0, 1]
            modelo.named_steps['classifier'].partial_fit(
                modelo.named_steps['tfidf'].transform(
                    modelo.named_steps['preprocessing'].transform(X)
                ),
                y,
                classes=clases
            )
        else:
            preprocessor = modelo.named_steps['preprocessing']
            vectorizer = modelo.named_steps['tfidf']
            clf_anterior = modelo.named_steps['classifier']

            X_preprocessed = preprocessor.transform(X)
            X_vect = vectorizer.transform(X_preprocessed)

            if modelo_id == "logistic_regression":
                from sklearn.linear_model import LogisticRegression
                params = clf_anterior.get_params()
                clf_nuevo = LogisticRegression(**params)
                clf_nuevo.fit(X_vect, y)

                modelo.named_steps['classifier'] = clf_nuevo

            elif modelo_id == "xgboost":
                from xgboost import XGBClassifier
                params = clf_anterior.get_params()
                n_estimators_prev = params.get('n_estimators', 70)
                clf_nuevo = XGBClassifier(**params)
                clf_nuevo.set_params(n_estimators=n_estimators_prev + 10)
                clf_nuevo.fit(X_vect, y)
                modelo.named_steps['classifier'] = clf_nuevo

            elif modelo_id == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                clf_nuevo = RandomForestClassifier(n_estimators=70, max_depth=None)
                clf_nuevo.fit(X_vect, y)
                modelo.named_steps['classifier'] = clf_nuevo

        dump(modelo, ruta)

        y_pred = modelo.predict(X)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

        return {
            "mensaje": f"Modelo '{modelo_id}' actualizado exitosamente",
            "metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4)
            }
        }

    except Exception as e:
        print("Error en /retrain:", str(e))
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
