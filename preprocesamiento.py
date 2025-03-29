from sklearn.base import BaseEstimator, TransformerMixin
import unicodedata
import re
import inflect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer


# Clase personalizada para preprocesar texto
# Esta clase se puede usar directamente en un pipeline de sklearn

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Inicializa recursos necesarios
        self.stop_words = set(stopwords.words('spanish'))  # Palabras vacías en español
        self.stemmer = LancasterStemmer()                  # Algoritmo de stemming
        self.lemmatizer = WordNetLemmatizer()              # Algoritmo de lematización
        self.p = inflect.engine()                          # Motor para convertir números a texto

    def fit(self, X, y=None):
        # Esta función es obligatoria para los transformadores de sklearn, aunque no se use
        return self

    def transform(self, X):
        # Aplica la limpieza y normalización de texto a cada entrada
        return [
            ' '.join(self.stem_and_lemmatize(self.clean_text(text)))
            for text in X
        ]

    def clean_text(self, text):
        
        # Limpia el texto aplicando las siguientes transformaciones:
        # - Tokenización
        # - Pasar a minúsculas
        # - Convertir números a palabras
        # - Eliminar signos de puntuación
        # - Normalizar caracteres unicode
        # - Eliminar stopwords
      
        words = word_tokenize(text)  # Divide en palabras
        words = [word.lower() for word in words]  # Minúsculas
        words = [
            self.p.number_to_words(word) if word.isdigit() else word
            for word in words
        ]
        words = [re.sub(r'[^\w\s]', '', word) for word in words]  # Eliminar signos
        words = [
            unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            for word in words
        ]
        words = [word for word in words if word not in self.stop_words and word.strip()]
        return words

    def stem_and_lemmatize(self, words):
        # Aplica stemming y lematización, y retorna ambos juntos
        stems = [self.stemmer.stem(word) for word in words]
        lemmas = [self.lemmatizer.lemmatize(word, pos='v') for word in words]
        return stems + lemmas

