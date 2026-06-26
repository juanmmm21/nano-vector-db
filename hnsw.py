import numpy as np
import logging
from typing import List, Tuple, Dict, Set, Union, Optional
from distance import cosine_distance, l2_distance, dot_product_distance

logger = logging.getLogger(__name__)

class HNSWNode:
    """
    Representacion de un nodo dentro del grafo multicapa HNSW.
    
    Cada nodo almacena su vector de características, su identificador único
    y las conexiones bidireccionales con otros nodos indexadas por nivel.
    """
    
    def __init__(self, node_id: Union[str, int], vector: np.ndarray, level: int) -> None:
        self.node_id = node_id
        self.vector = np.asarray(vector, dtype=np.float32)
        self.level = level
        # Un diccionario que mapea cada nivel en el que existe el nodo a su conjunto de vecinos
        self.neighbors: Dict[int, Set["HNSWNode"]] = {l: set() for l in range(level + 1)}

    def __repr__(self) -> str:
        return f"HNSWNode(id={self.node_id}, level={self.level})"


class HNSWIndex:
    """
    Indice aproximado de vecinos mas cercanos basado en grafos jerarquicos (HNSW).
    
    Optimiza la busqueda de vecinos mas cercanos reduciendo la complejidad
    temporal de O(N) a O(log N) mediante una estructura de saltos similar a una Skip List.
    """
    
    def __init__(
        self,
        dimension: int,
        metric: str = "cosine",
        M: int = 16,
        efConstruction: int = 64,
        efSearch: int = 32
    ) -> None:
        """
        Args:
            dimension: Dimension de los vectores a indexar.
            metric: Metrica de distancia ('cosine', 'l2', 'dot_product').
            M: Numero maximo de enlaces bidireccionales por nodo en cada capa > 0.
            efConstruction: Profundidad de busqueda de candidatos durante la construccion.
            efSearch: Profundidad de busqueda de candidatos durante consultas de busqueda.
        """
        self.dimension = dimension
        self.metric = metric
        self.M = M
        self.M0 = 2 * M  # Capa 0 suele admitir el doble de conexiones para mejorar la conectividad base
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        
        # Factor de normalizacion para la distribucion del nivel de los nuevos nodos.
        # Determina la probabilidad de que un nodo ascienda a capas superiores.
        self.mL = 1.0 / np.log(M)
        
        # Seleccion de la funcion de distancia
        if metric == "cosine":
            self.distance_fn = cosine_distance
        elif metric == "l2":
            self.distance_fn = l2_distance
        elif metric == "dot_product":
            self.distance_fn = dot_product_distance
        else:
            raise ValueError(f"Metrica '{metric}' no soportada. Usar 'cosine', 'l2' o 'dot_product'.")
            
        self.enter_node: Optional[HNSWNode] = None
        self.max_level: int = -1
        self.nodes: Dict[Union[str, int], HNSWNode] = {}
        
    def _generate_random_level(self) -> int:
        """
        Determina de forma probabilistica en que nivel maximo residira el nuevo nodo.
        Utiliza una distribucion exponencial decayente para simular la Skip List.
        """
        # Añadimos un pequeño epsilon para evitar el log(0)
        r = np.random.uniform(1e-9, 1.0)
        return int(-np.log(r) * self.mL)

    def search_layer(
        self,
        q: np.ndarray,
        enter_points: List[HNSWNode],
        ef: int,
        level: int
    ) -> List[Tuple[float, HNSWNode]]:
        """
        Busca los vecinos mas cercanos a un vector consulta 'q' dentro de un nivel especifico.
        Implementa el algoritmo de busqueda acotado por 'ef'.
        """
        # Mantendremos registro de los nodos visitados por sus IDs para evitar bucles infinitos
        visited: Set[Union[str, int]] = set(node.node_id for node in enter_points)
        
        # Evaluamos distancias iniciales de los puntos de entrada
        candidates: List[Tuple[float, HNSWNode]] = []
        for node in enter_points:
            dist = self.distance_fn(q, node.vector)
            candidates.append((dist, node))
            
        # Ordenamos los candidatos de menor a mayor distancia
        candidates.sort(key=lambda x: x[0])
        
        # W representa nuestra lista dinamica de los mejores resultados encontrados hasta ahora (tamaño maximo ef)
        W = list(candidates)
        
        while len(candidates) > 0:
            # Extraemos el candidato mas cercano actual
            curr_dist, curr_node = candidates.pop(0)
            
            # Si el elemento mas cercano a evaluar esta mas lejos que el peor elemento en W, detenemos la busqueda
            furthest_in_W = W[-1][0]
            if curr_dist > furthest_in_W:
                break
                
            # Evaluamos los vecinos del nodo actual en el nivel dado
            neighbors = curr_node.neighbors.get(level, set())
            for neighbor in neighbors:
                if neighbor.node_id not in visited:
                    visited.add(neighbor.node_id)
                    
                    dist = self.distance_fn(q, neighbor.vector)
                    furthest_in_W = W[-1][0]
                    
                    # Si el vecino es mas cercano que el peor en W, o aun no alcanzamos la capacidad ef
                    if dist < furthest_in_W or len(W) < ef:
                        candidates.append((dist, neighbor))
                        candidates.sort(key=lambda x: x[0])
                        
                        W.append((dist, neighbor))
                        W.sort(key=lambda x: x[0])
                        
                        # Acotamos W al tamaño ef
                        if len(W) > ef:
                            W.pop()
                            
        return W

    def add_node(self, node_id: Union[str, int], vector: Union[List[float], np.ndarray]) -> None:
        """
        Inserta un nuevo nodo en el grafo HNSW estableciendo conexiones bidireccionales podadas.
        """
        vec_arr = np.asarray(vector, dtype=np.float32)
        if vec_arr.shape[0] != self.dimension:
            raise ValueError(f"Dimension del vector ({vec_arr.shape[0]}) no coincide con la del indice ({self.dimension})")
            
        if node_id in self.nodes:
            # Si ya existe, podriamos actualizarlo, pero por simplicidad de base de datos lanzamos error
            raise ValueError(f"El nodo con ID '{node_id}' ya existe en el indice.")
            
        # 1. Determinamos nivel del nuevo nodo
        level = self._generate_random_level()
        new_node = HNSWNode(node_id, vec_arr, level)
        self.nodes[node_id] = new_node
        
        # Si el indice estaba vacio, este nodo es el nuevo punto de entrada principal
        if self.enter_node is None:
            self.enter_node = new_node
            self.max_level = level
            return
            
        # 2. Busqueda descendente rapida (Greedy Search) desde max_level hasta level + 1
        # El objetivo es encontrar el punto de entrada optimo al nivel donde se insertara el nodo
        curr_node = self.enter_node
        curr_dist = self.distance_fn(vec_arr, curr_node.vector)
        
        for l in range(self.max_level, level, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in curr_node.neighbors.get(l, set()):
                    dist = self.distance_fn(vec_arr, neighbor.vector)
                    if dist < curr_dist:
                        curr_dist = dist
                        curr_node = neighbor
                        changed = True
                        
        # 3. Insercion bidireccional desde el nivel de insercion (minimo entre el nivel del nodo y el maximo del grafo) hasta 0
        enter_points = [curr_node]
        start_level = min(self.max_level, level)
        
        for l in range(start_level, -1, -1):
            # Buscamos efConstruction candidatos cercanos en esta capa
            candidates_W = self.search_layer(vec_arr, enter_points, self.efConstruction, l)
            
            # Determinamos cuantas conexiones maximas permitimos en esta capa
            max_conn = self.M if l > 0 else self.M0
            
            # Seleccionamos los M vecinos mas cercanos (heuristica simple de proximidad)
            neighbors_to_connect = candidates_W[:max_conn]
            
            # Creamos los enlaces bidireccionales
            new_node.neighbors[l] = set(neighbor for _, neighbor in neighbors_to_connect)
            for _, neighbor in neighbors_to_connect:
                neighbor.neighbors[l].add(new_node)
                
                # Si el vecino excede su cuota maxima de conexiones en este nivel, debemos podar las peores
                if len(neighbor.neighbors[l]) > max_conn:
                    self._prune_connections(neighbor, l, max_conn)
                    
            # Los candidatos encontrados en esta capa sirven como puntos de entrada para la capa inferior
            enter_points = [node for _, node in candidates_W]
            
        # 4. Si el nuevo nodo supera el nivel maximo registrado, actualizamos el punto de entrada global
        if level > self.max_level:
            self.max_level = level
            self.enter_node = new_node

    def _prune_connections(self, node: HNSWNode, level: int, max_conn: int) -> None:
        """
        Poda las conexiones de un nodo en un nivel dado, conservando unicamente las 'max_conn' mas cercanas.
        """
        connections = list(node.neighbors[level])
        conn_dists = []
        for conn in connections:
            dist = self.distance_fn(node.vector, conn.vector)
            conn_dists.append((dist, conn))
            
        # Ordenamos por cercania
        conn_dists.sort(key=lambda x: x[0])
        
        # Sobreescribimos con el conjunto recortado de vecinos mas proximos
        node.neighbors[level] = set(conn for _, conn in conn_dists[:max_conn])

    def query(self, query_vector: Union[List[float], np.ndarray], k: int = 5) -> List[Tuple[float, Union[str, int]]]:
        """
        Realiza la busqueda aproximada de los K vecinos mas cercanos.
        """
        if self.enter_node is None:
            return []
            
        vec_arr = np.asarray(query_vector, dtype=np.float32)
        
        # 1. Busqueda Greedy rapida hasta la capa 1
        curr_node = self.enter_node
        curr_dist = self.distance_fn(vec_arr, curr_node.vector)
        
        for l in range(self.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in curr_node.neighbors.get(l, set()):
                    dist = self.distance_fn(vec_arr, neighbor.vector)
                    if dist < curr_dist:
                        curr_dist = dist
                        curr_node = neighbor
                        changed = True
                        
        # 2. Busqueda exhaustiva local en capa 0 usando efSearch
        candidates_W = self.search_layer(vec_arr, [curr_node], self.efSearch, 0)
        
        # 3. Retornamos los K mejores ordenados en tupla (distancia, node_id)
        return [(dist, node.node_id) for dist, node in candidates_W[:k]]
