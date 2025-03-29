
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Estructuras de datos para validar las entradas de los endpoints
from structure import DataModel, ReentrenamientoModel

# Para guardar/cargar modelos
from joblib import load, dump

# Métricas de evaluación
from sklearn.metrics import precision_score, recall_score, f1_score

# Para dividir los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split

# Componentes del pipeline de ML
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Preprocesador personalizado creado por ti
from preprocesamiento import Preprocessor

# Utilidades
import os
import pandas as pd


# Inicialización de la aplicación FastAPI

app = FastAPI(title="Clasificación de noticias falsas")


# Configuración CORS

# Esto permite que cualquier frontend (React, HTML, etc.) pueda comunicarse con la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite peticiones desde cualquier origen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Diccionario de modelos disponibles

# Se utiliza para mapear los nombres de modelos a sus rutas en disco
MODELOS = {
    "naive_bayes": "modelos/naive_bayes.joblib",
    "modelo1": "modelos/modelo1.joblib",  # Agregado para posibles modelos de tus compañeros
    "modelo2": "modelos/modelo2.joblib"
}


# Endpoint raíz de prueba

@app.get("/")
def root():
    return {
        "mensaje": "API disponible",
        "modelos_disponibles": list(MODELOS.keys())
    }


# Endpoint de predicción

@app.post("/predict")
def predecir(data: DataModel):
    modelo_id = data.modelo

    # Validación de modelo
    if modelo_id not in MODELOS:
        raise HTTPException(status_code=400, detail="Modelo no soportado")

    ruta = MODELOS[modelo_id]
    if not os.path.exists(ruta):
        raise HTTPException(status_code=404, detail="Modelo no encontrado en el servidor")

    # Cargar el modelo desde disco
    modelo = load(ruta)

    # Combinar descripción + título en un solo texto
    textos = [desc + " " + tit for desc, tit in zip(data.Descripcion, data.Titulo)]

    # Realizar predicción y probabilidades
    pred = modelo.predict(textos).tolist()
    prob = modelo.predict_proba(textos).tolist()

    return {
        "modelo_usado": modelo_id,
        "predicciones": pred,
        "probabilidades": prob
    }


# Endpoint de reentrenamiento

@app.post("/retrain")
def reentrenar(data: ReentrenamientoModel):
    modelo_id = data.modelo

    # Validar que el modelo esté registrado
    if modelo_id not in MODELOS:
        raise HTTPException(status_code=400, detail="Modelo no soportado")

    ruta = MODELOS[modelo_id]

    try:
        # Preparar los textos unificando descripción + título
        textos = [desc + " " + tit for desc, tit in zip(data.Descripcion, data.Titulo)]
        df = pd.DataFrame({'texto': textos, 'Label': data.Label})

        # Dividir en entrenamiento y prueba con estratificación
        X_train, X_test, y_train, y_test = train_test_split(
            df['texto'], df['Label'], test_size=0.2, random_state=42, stratify=df['Label']
        )

        # Reconstruir desde cero el pipeline con el preprocesamiento, TF-IDF y clasificador
        nuevo_pipeline = Pipeline([
            ('preprocessing', Preprocessor()),
            ('tfidf', TfidfVectorizer()),
            ('clf', MultinomialNB())
        ])

        # Entrenar el pipeline
        nuevo_pipeline.fit(X_train, y_train)

        # Guardar el modelo actualizado
        dump(nuevo_pipeline, ruta)

        # Evaluar métricas sobre conjunto de prueba
        y_pred = nuevo_pipeline.predict(X_test)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        return {
            "mensaje": f"Modelo '{modelo_id}' reentrenado y guardado exitosamente",
            "metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4)
            }
        }

    except Exception as e:
        # Si ocurre un error, mostrarlo en consola y enviar mensaje de error al cliente
        print("Error en /retrain:", str(e))
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")




