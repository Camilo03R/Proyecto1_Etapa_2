from pydantic import BaseModel
from typing import List


# Modelo de datos para predicción de noticias
# Se usa en el endpoint POST /predict

class DataModel(BaseModel):
    modelo: str  # Nombre del modelo a utilizar (por ejemplo: "naive_bayes")
    Descripcion: List[str]  # Lista de descripciones de las noticias
    Titulo: List[str]       # Lista de títulos de las noticias


# Modelo de datos para reentrenamiento del modelo
# Se usa en el endpoint POST /retrain

class ReentrenamientoModel(BaseModel):
    modelo: str  # Nombre del modelo a reentrenar (debe estar en el diccionario de modelos)
    Descripcion: List[str]  # Lista de descripciones
    Titulo: List[str]       # Lista de títulos
    Label: List[int]        # Etiquetas correspondientes (0 = falsa, 1 = verdadera)
