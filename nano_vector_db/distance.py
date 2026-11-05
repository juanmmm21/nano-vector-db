import numpy as np
from typing import Union, List

# Definimos un tipo flexible para aceptar tanto listas nativas como arrays de NumPy
VectorLike = Union[List[float], np.ndarray]

def cosine_distance(u: VectorLike, v: VectorLike) -> float:
    """
    Calcula la distancia de coseno entre dos vectores.
    
    Se define matematicamente como: 1.0 - CosineSimilarity(u, v).
    Valores cercanos a 0.0 indican alta similitud. Valores cercanos a 2.0 indican oposicion.
    """
    arr_u = np.asarray(u, dtype=np.float32)
    arr_v = np.asarray(v, dtype=np.float32)
    
    norm_u = np.linalg.norm(arr_u)
    norm_v = np.linalg.norm(arr_v)
    
    # Prevenimos division por cero en vectores nulos (sin magnitud)
    if norm_u < 1e-9 or norm_v < 1e-9:
        return 1.0
        
    dot_product = np.dot(arr_u, arr_v)
    similarity = dot_product / (norm_u * norm_v)
    
    # Acotamos la similitud por posibles imprecisiones decimales fuera del rango [-1, 1]
    similarity = np.clip(similarity, -1.0, 1.0)
    return float(1.0 - similarity)

def l2_distance(u: VectorLike, v: VectorLike) -> float:
    """
    Calcula la distancia euclidea (L2) entre dos vectores.
    
    Es util para comparar magnitudes absolutas ademas de la direccion semantica.
    """
    arr_u = np.asarray(u, dtype=np.float32)
    arr_v = np.asarray(v, dtype=np.float32)
    return float(np.linalg.norm(arr_u - arr_v))

def dot_product_distance(u: VectorLike, v: VectorLike) -> float:
    """
    Calcula el producto escalar invertido como metrica de distancia.
    
    Ideal si los vectores ya se encuentran previamente normalizados L2 (como es el caso
    de los embeddings generados por nuestro contrastive-embedding-trainer).
    """
    arr_u = np.asarray(u, dtype=np.float32)
    arr_v = np.asarray(v, dtype=np.float32)
    
    # Multiplicamos por -1.0 para que el optimizador busque minimizar el valor,
    # puesto que a mayor producto escalar (mas similar) menor sera el valor resultante.
    return float(-np.dot(arr_u, arr_v))
