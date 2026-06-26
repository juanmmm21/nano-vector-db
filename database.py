import os
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
from distance import cosine_distance, l2_distance, dot_product_distance
from hnsw import HNSWIndex

logger = logging.getLogger(__name__)

class NanoVectorDB:
    """
    Motor principal de la base de datos vectorial NanoVectorDB.
    
    Gestiona el almacenamiento de vectores, metadatos asociados y realiza
    busquedas semanticas exactas (Flat) o aproximadas (HNSW) con soporte
    de filtrado avanzado por metadatos (MongoDB-style).
    """
    
    def __init__(
        self,
        dimension: int,
        metric: str = "cosine",
        index_type: str = "hnsw",
        M: int = 16,
        efConstruction: int = 64,
        efSearch: int = 32
    ) -> None:
        """
        Args:
            dimension: Dimension de los vectores que almacenara la base de datos.
            metric: Metrica de distancia ('cosine', 'l2', 'dot_product').
            index_type: Metodo de indexacion ('hnsw' o 'flat').
            M: Hiperparametro de conexiones maximas en HNSW.
            efConstruction: Hiperparametro de profundidad de busqueda en HNSW (construccion).
            efSearch: Hiperparametro de profundidad de busqueda en HNSW (busqueda).
        """
        self.dimension = dimension
        self.metric = metric
        self.index_type = index_type.lower()
        
        self.vectors: Dict[Union[str, int], np.ndarray] = {}
        self.metadata: Dict[Union[str, int], Dict[str, Any]] = {}
        
        # Seleccion de la funcion de distancia local para busquedas Flat
        if metric == "cosine":
            self.distance_fn = cosine_distance
        elif metric == "l2":
            self.distance_fn = l2_distance
        elif metric == "dot_product":
            self.distance_fn = dot_product_distance
        else:
            raise ValueError(f"Metrica '{metric}' no soportada.")
            
        # Inicializacion del indice HNSW si procede
        if self.index_type == "hnsw":
            self.index = HNSWIndex(
                dimension=dimension,
                metric=metric,
                M=M,
                efConstruction=efConstruction,
                efSearch=efSearch
            )
        elif self.index_type == "flat":
            self.index = None
        else:
            raise ValueError(f"Tipo de indice '{index_type}' invalido. Usar 'hnsw' o 'flat'.")
            
    def insert(
        self,
        id: Union[str, int],
        vector: Union[List[float], np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Inserta un vector y sus metadatos correspondientes en la base de datos.
        """
        vec_arr = np.asarray(vector, dtype=np.float32)
        if vec_arr.shape[0] != self.dimension:
            raise ValueError(f"El vector tiene dimension {vec_arr.shape[0]}, pero la DB espera {self.dimension}.")
            
        if id in self.vectors:
            raise ValueError(f"El ID '{id}' ya existe en la base de datos.")
            
        self.vectors[id] = vec_arr
        self.metadata[id] = metadata if metadata is not None else {}
        
        # Si HNSW esta habilitado, agregamos el nodo al indice
        if self.index_type == "hnsw" and self.index is not None:
            self.index.add_node(id, vec_arr)
            
    def _evaluate_filter(self, doc_metadata: Dict[str, Any], query_filter: Dict[str, Any]) -> bool:
        """
        Evalua de forma recursiva si los metadatos de un documento cumplen con las
        condiciones de filtrado especificadas (estilo operadores de MongoDB).
        """
        for field, condition in query_filter.items():
            if field not in doc_metadata:
                return False
                
            val = doc_metadata[field]
            
            if isinstance(condition, dict):
                # Evaluamos operadores avanzados
                for op, op_val in condition.items():
                    op_lower = op.lower()
                    if op_lower == "$eq":
                        if val != op_val: return False
                    elif op_lower == "$ne":
                        if val == op_val: return False
                    elif op_lower == "$gt":
                        if not (val > op_val): return False
                    elif op_lower == "$gte":
                        if not (val >= op_val): return False
                    elif op_lower == "$lt":
                        if not (val < op_val): return False
                    elif op_lower == "$lte":
                        if not (val <= op_val): return False
                    elif op_lower == "$in":
                        if not isinstance(op_val, list) or val not in op_val: return False
                    elif op_lower == "$nin":
                        if isinstance(op_val, list) and val in op_val: return False
                    else:
                        raise ValueError(f"Operador de filtro '{op}' no soportado.")
            else:
                # Comparacion directa de igualdad implicita
                if val != condition:
                    return False
        return True

    def query(
        self,
        vector: Union[List[float], np.ndarray],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Realiza la busqueda de los vecinos mas cercanos.
        
        Soporta pre-filtrado en el modo exacto (Flat) y post-filtrado
        en el modo aproximado (HNSW) para maximizar la velocidad de busqueda.
        
        Returns:
            Lista de diccionarios, conteniendo 'id', 'distance', 'metadata' y 'vector'.
        """
        vec_arr = np.asarray(vector, dtype=np.float32)
        
        # Caso 1: Busqueda Exacta (Flat). Optima para datasets pequeños o cuando el filtro
        # es muy restrictivo y se requiere exactitud absoluta (pre-filtrado).
        if self.index_type == "flat" or self.index is None:
            return self._flat_query(vec_arr, top_k, filter)
            
        # Caso 2: Busqueda Aproximada (HNSW) con Post-filtrado.
        # Obtenemos un conjunto amplio de candidatos aproximados mediante el grafo.
        # Usamos efSearch para tener un colchón de candidatos en caso de descarte por metadatos.
        ef = max(self.index.efSearch, top_k * 4)
        raw_candidates = self.index.query(vec_arr, k=ef)
        
        results = []
        for dist, id in raw_candidates:
            doc_metadata = self.metadata.get(id, {})
            # Si hay filtro, lo evaluamos antes de incluir el resultado
            if filter is not None:
                if not self._evaluate_filter(doc_metadata, filter):
                    continue
            results.append({
                "id": id,
                "distance": dist,
                "metadata": doc_metadata,
                "vector": self.vectors[id].tolist()
            })
            if len(results) == top_k:
                break
                
        # Si la busqueda aproximada se quedo sin candidatos por un filtro muy restrictivo,
        # hacemos un fallback silencioso a Flat exacto para garantizar que devolvemos los resultados.
        if len(results) < top_k and filter is not None:
            logger.info("El post-filtrado HNSW devolvio menos de top_k candidatos. Ejecutando fallback a Flat.")
            return self._flat_query(vec_arr, top_k, filter)
            
        return results

    def _flat_query(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta una busqueda secuencial exacta (fuerza bruta) con pre-filtrado.
        """
        candidates: List[Tuple[float, Union[str, int]]] = []
        
        for id, vec in self.vectors.items():
            doc_metadata = self.metadata.get(id, {})
            # Pre-filtrado: si no cumple las condiciones, se descarta antes de computar distancia
            if filter is not None:
                if not self._evaluate_filter(doc_metadata, filter):
                    continue
                    
            dist = self.distance_fn(query_vector, vec)
            candidates.append((dist, id))
            
        # Ordenamos de menor a mayor distancia
        candidates.sort(key=lambda x: x[0])
        
        # Estructuramos el top_k de salida
        results = []
        for dist, id in candidates[:top_k]:
            results.append({
                "id": id,
                "distance": dist,
                "metadata": self.metadata[id],
                "vector": self.vectors[id].tolist()
            })
        return results

    def save(self, filepath: str) -> None:
        """
        Persiste el estado completo de la base de datos (nodos, indices y metadatos)
        en disco en formato binario estructurado utilizando pickle.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        try:
            with open(filepath, "wb") as f:
                pickle.dump(self, f)
            logger.info(f"Base de datos guardada exitosamente en {filepath}")
        except Exception as e:
            logger.error(f"Error al guardar la base de datos en {filepath}: {str(e)}")
            raise

    @classmethod
    def load(cls, filepath: str) -> "NanoVectorDB":
        """
        Carga una base de datos persistida previamente en formato binario.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontro el archivo de base de datos en: {filepath}")
            
        try:
            with open(filepath, "rb") as f:
                db = pickle.load(f)
            if not isinstance(db, cls):
                raise TypeError(f"El archivo cargado no es una instancia valida de {cls.__name__}.")
            logger.info(f"Base de datos cargada exitosamente desde {filepath}")
            return db
        except Exception as e:
            logger.error(f"Error al cargar la base de datos desde {filepath}: {str(e)}")
            raise
