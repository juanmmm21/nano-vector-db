import unittest
import numpy as np
import tempfile
import shutil
import os
from typing import Dict, Any

from distance import cosine_distance, l2_distance, dot_product_distance
from hnsw import HNSWIndex
from database import NanoVectorDB


class TestNanoVectorDB(unittest.TestCase):
    """
    Suite de pruebas unitarias para verificar la exactitud del motor
    de calculo, el grafo HNSW y la base de datos NanoVectorDB.
    """

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_db.pkl")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_distance_metrics(self) -> None:
        """
        Verifica la correctitud matematica de las funciones de distancia.
        """
        u = [1.0, 0.0, 0.0]
        v = [1.0, 0.0, 0.0]
        w = [0.0, 1.0, 0.0]
        z = [-1.0, 0.0, 0.0]
        
        # 1. Coseno
        self.assertAlmostEqual(cosine_distance(u, v), 0.0)
        self.assertAlmostEqual(cosine_distance(u, w), 1.0)
        self.assertAlmostEqual(cosine_distance(u, z), 2.0)
        
        # 2. L2 (Euclidea)
        self.assertAlmostEqual(l2_distance(u, v), 0.0)
        self.assertAlmostEqual(l2_distance(u, w), np.sqrt(2))
        
        # 3. Producto Escalar
        # Si estan normalizados, u.v = 1.0 -> distancia = -1.0
        self.assertAlmostEqual(dot_product_distance(u, v), -1.0)
        self.assertAlmostEqual(dot_product_distance(u, z), 1.0)

    def test_database_insert_validation(self) -> None:
        """
        Verifica las validaciones de insercion de vectores en NanoVectorDB.
        """
        db = NanoVectorDB(dimension=3, index_type="flat")
        
        # Insercion exitosa
        db.insert(id="doc1", vector=[1.0, 2.0, 3.0], metadata={"category": "test"})
        self.assertEqual(len(db.vectors), 1)
        
        # Insercion con dimension erronea
        with self.assertRaises(ValueError):
            db.insert(id="doc2", vector=[1.0, 2.0], metadata={})
            
        # Insercion con ID duplicado
        with self.assertRaises(ValueError):
            db.insert(id="doc1", vector=[2.0, 3.0, 4.0], metadata={})

    def test_database_flat_query(self) -> None:
        """
        Verifica la busqueda exacta (Flat) y ordenamiento correcto de distancias.
        """
        db = NanoVectorDB(dimension=2, index_type="flat", metric="cosine")
        db.insert(id="A", vector=[1.0, 0.0], metadata={"name": "A"})
        db.insert(id="B", vector=[0.9, 0.1], metadata={"name": "B"})
        db.insert(id="C", vector=[0.0, 1.0], metadata={"name": "C"})
        
        # Consultamos buscando el mas similar a [1.0, 0.0]
        results = db.query(vector=[1.0, 0.0], top_k=2)
        
        self.assertEqual(len(results), 2)
        # El primero debe ser A (distancia 0.0)
        self.assertEqual(results[0]["id"], "A")
        self.assertAlmostEqual(results[0]["distance"], 0.0, places=5)
        # El segundo debe ser B
        self.assertEqual(results[1]["id"], "B")

    def test_database_hnsw_query(self) -> None:
        """
        Verifica la exactitud de busqueda en modo HNSW comparada con Flat.
        """
        # Creamos dos bases de datos idénticas con distintos indices
        db_flat = NanoVectorDB(dimension=8, index_type="flat", metric="l2")
        db_hnsw = NanoVectorDB(dimension=8, index_type="hnsw", metric="l2", M=8, efConstruction=32, efSearch=16)
        
        # Insertamos un lote de vectores aleatorios deterministas (semilla fija)
        np.random.seed(42)
        for i in range(100):
            vec = np.random.randn(8)
            db_flat.insert(id=i, vector=vec)
            db_hnsw.insert(id=i, vector=vec)
            
        # Ejecutamos consultas aleatorias y medimos exactitud (coincidencia de vecinos mas cercanos)
        for _ in range(5):
            query_vec = np.random.randn(8)
            res_flat = db_flat.query(query_vec, top_k=3)
            res_hnsw = db_hnsw.query(query_vec, top_k=3)
            
            # Verificamos que al menos el vecino mas cercano absoluto coincida (HNSW suele tener recall > 95%)
            self.assertEqual(res_flat[0]["id"], res_hnsw[0]["id"])

    def test_metadata_filtering_operators(self) -> None:
        """
        Prueba los operadores avanzados de filtrado estilo MongoDB ($eq, $ne, $in, $nin, $gt, $lt, etc).
        """
        db = NanoVectorDB(dimension=2, index_type="flat")
        
        db.insert(id=1, vector=[1.0, 0.0], metadata={"age": 25, "tags": "nlp", "status": "active"})
        db.insert(id=2, vector=[0.0, 1.0], metadata={"age": 30, "tags": "cv", "status": "active"})
        db.insert(id=3, vector=[0.5, 0.5], metadata={"age": 45, "tags": "nlp", "status": "pending"})
        
        # 1. Filtro simple (igualdad implicita)
        res = db.query(vector=[1.0, 0.0], top_k=3, filter={"tags": "nlp"})
        ids = [r["id"] for r in res]
        self.assertEqual(set(ids), {1, 3})
        
        # 2. Operador $eq y $ne
        res = db.query(vector=[1.0, 0.0], top_k=3, filter={"status": {"$ne": "active"}})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["id"], 3)
        
        # 3. Operadores numericos $gt y $lte
        res = db.query(vector=[1.0, 0.0], top_k=3, filter={"age": {"$gt": 28, "$lte": 40}})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["id"], 2)
        
        # 4. Operadores de pertenencia $in y $nin
        res = db.query(vector=[1.0, 0.0], top_k=3, filter={"tags": {"$in": ["nlp", "cv"]}})
        self.assertEqual(len(res), 3)
        
        res = db.query(vector=[1.0, 0.0], top_k=3, filter={"tags": {"$nin": ["cv"]}})
        ids = [r["id"] for r in res]
        self.assertEqual(set(ids), {1, 3})

    def test_database_hnsw_fallback_to_flat(self) -> None:
        """
        Verifica que HNSW realice fallback a Flat si el post-filtrado
        elimina demasiados candidatos y no alcanza la cuota 'top_k'.
        """
        db = NanoVectorDB(dimension=2, index_type="hnsw", metric="l2", efSearch=2)
        db.insert(id=1, vector=[1.0, 0.0], metadata={"type": "A"})
        db.insert(id=2, vector=[0.9, 0.1], metadata={"type": "B"})
        db.insert(id=3, vector=[0.8, 0.2], metadata={"type": "B"})
        
        # Buscamos mas cercano con filtro restrictivo {"type": "B"}.
        # Si efSearch=2, el grafo HNSW podria devolver solo los IDs 1 y 2 en su paso aproximado.
        # Al filtrar por "B", quedaria solo el ID 2 (largo 1, menor que top_k=2).
        # El fallback entra y recupera secuencialmente el ID 3 tambien.
        res = db.query(vector=[1.0, 0.0], top_k=2, filter={"type": "B"})
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["id"], 2)
        self.assertEqual(res[1]["id"], 3)

    def test_database_serialization(self) -> None:
        """
        Prueba el guardado y carga exitosa de la base de datos en disco.
        """
        db = NanoVectorDB(dimension=3, index_type="hnsw", metric="cosine")
        db.insert(id="doc1", vector=[1.0, 0.0, 0.0], metadata={"tag": "red"})
        db.insert(id="doc2", vector=[0.0, 1.0, 0.0], metadata={"tag": "green"})
        
        # Guardamos a disco
        db.save(self.db_path)
        self.assertTrue(os.path.exists(self.db_path))
        
        # Cargamos en una nueva variable
        loaded_db = NanoVectorDB.load(self.db_path)
        
        self.assertEqual(loaded_db.dimension, 3)
        self.assertEqual(loaded_db.metric, "cosine")
        self.assertEqual(loaded_db.index_type, "hnsw")
        self.assertEqual(len(loaded_db.vectors), 2)
        
        # Verificamos que las busquedas sobre el modelo cargado devuelvan resultados validos
        res = loaded_db.query(vector=[1.0, 0.0, 0.0], top_k=1)
        self.assertEqual(res[0]["id"], "doc1")


if __name__ == "__main__":
    unittest.main()
