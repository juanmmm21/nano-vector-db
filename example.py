import os
import time
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from database import NanoVectorDB

# Intentamos importar librerias de Deep Learning para habilitar la busqueda semantica real (Interlinking)
SEMAN_SEARCH_ENABLED = False
tokenizer = None
model = None
torch = None

MODEL_PATH = "../contrastive-embedding-trainer/model_output"

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    if os.path.exists(MODEL_PATH) and os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModel.from_pretrained(MODEL_PATH)
        model.eval()
        SEMAN_SEARCH_ENABLED = True
except ImportError:
    pass


def get_embedding(text: str) -> np.ndarray:
    """
    Genera el embedding L2 normalizado de una frase.
    
    Si el modulo 'contrastive-embedding-trainer' esta disponible con sus pesos,
    genera un embedding semantico real. En caso contrario, genera un vector determinista
    basado en el hash del texto como simulacion matematica.
    """
    if SEMAN_SEARCH_ENABLED and tokenizer is not None and model is not None and torch is not None:
        with torch.no_grad():
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )
            outputs = model(**inputs)
            
            # Aplicamos Mean Pooling sobre los estados ocultos
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            embedding = sum_embeddings / sum_mask
            # Normalizacion L2 para compatibilidad con similitud de coseno
            normalized = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return normalized.squeeze(0).cpu().numpy()
    else:
        # Generador de embeddings determinista basado en semillas fijas del hash del texto.
        # Permite probar la base de datos sin descargar modelos ni requerir GPUs.
        state = np.random.RandomState(abs(hash(text)) % (2**32))
        vec = state.normal(0.0, 1.0, 768)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-9 else vec


def main() -> None:
    print("==================================================")
    print("       Demostracion de NanoVectorDB y HNSW        ")
    print("==================================================")
    
    # 1. Definicion de la base de datos
    dimension = 768
    metric = "cosine"
    
    if SEMAN_SEARCH_ENABLED:
        print(f"Cargado el modelo local desde '{MODEL_PATH}'.")
        print("Busqueda semantica real HABILITADA.")
    else:
        print("Modelos locales o librerias (torch/transformers) no detectados.")
        print("Generador determinista de simulacion vectorial HABILITADO.")
        
    print("\nInicializando base de datos en modo HNSW...")
    # Parametros moderados para grafos pequeños (M=16, efConstruction=64, efSearch=32)
    db = NanoVectorDB(
        dimension=dimension,
        metric=metric,
        index_type="hnsw",
        M=16,
        efConstruction=64,
        efSearch=32
    )
    
    # Corpus de oraciones para poblar la base de datos
    documents = [
        # Astronomia
        ("La astronomia nos permite estudiar las estrellas y galaxias distantes.", {"category": "astronomia", "complexity": "alta"}),
        ("El cosmos es un lugar misterioso y en constante expansion acelerada.", {"category": "astronomia", "complexity": "media"}),
        ("La exploracion espacial por satelites aporta datos clave del sistema solar.", {"category": "astronomia", "complexity": "baja"}),
        
        # Programacion
        ("Aprender a programar en Python abre muchas puertas en desarrollo web y ciencia de datos.", {"category": "programacion", "complexity": "baja"}),
        ("La optimizacion de consultas SQL mejora drasticamente el rendimiento de las aplicaciones.", {"category": "programacion", "complexity": "alta"}),
        ("El codigo limpio y refactorizado reduce la deuda tecnica de los proyectos de software.", {"category": "programacion", "complexity": "media"}),
        
        # Cocina
        ("Una receta tradicional de paella requiere ingredientes frescos de mar o tierra.", {"category": "cocina", "complexity": "media"}),
        ("El pan de masa madre se fermenta de forma natural y requiere paciencia.", {"category": "cocina", "complexity": "media"}),
        ("Para freir patatas perfectas se recomienda usar aceite de oliva a fuego medio.", {"category": "cocina", "complexity": "baja"})
    ]
    
    print("\nInsertando y vectorizando documentos en el indice HNSW...")
    for idx, (text, meta) in enumerate(documents, start=1):
        vec = get_embedding(text)
        meta["text"] = text  # Almacenamos el texto en los metadatos para visualizar el resultado
        db.insert(id=idx, vector=vec, metadata=meta)
        print(f"Indexado [{idx}]: '{text[:50]}...' | Categoria: {meta['category']}")
        
    # 2. Consultas semanticas
    queries = [
        ("Como preparar una comida tradicional española", {"category": "cocina"}),
        ("El ciclo de vida de los sistemas de software y algoritmos", {"category": "programacion"}),
        ("Observacion de constelaciones celestes y planetas", None)
    ]
    
    print("\n" + "="*50)
    print(" Ejecutando consultas de busqueda vectorial y filtros ")
    print("="*50)
    
    for q_text, q_filter in queries:
        print(f"\nConsulta: '{q_text}'")
        if q_filter:
            print(f"Filtro aplicado: {q_filter}")
            
        q_vec = get_embedding(q_text)
        
        # Realizamos la busqueda aproximada en HNSW
        start_time = time.perf_counter()
        results = db.query(q_vec, top_k=2, filter=q_filter)
        exec_time = (time.perf_counter() - start_time) * 1000
        
        print(f"Resultados (tiempo de busqueda: {exec_time:.4f} ms):")
        for r in results:
            print(f"  - ID: {r['id']} | Distancia: {r['distance']:.4f} | '{r['metadata']['text']}'")

    # 3. Comparativa de velocidad/exactitud: HNSW vs Flat (Fuerza Bruta)
    print("\n" + "="*50)
    print(" Comparativa tecnica: HNSW vs Flat (Fuerza Bruta) ")
    print("="*50)
    
    # Creamos un indice exacto Flat con los mismos datos
    db_flat = NanoVectorDB(dimension=dimension, metric=metric, index_type="flat")
    for idx, vec in db.vectors.items():
        db_flat.insert(id=idx, vector=vec, metadata=db.metadata[idx])
        
    test_q_text = "Quiero programar algoritmos limpios en Python"
    test_q_vec = get_embedding(test_q_text)
    
    # Consulta en Flat
    t0 = time.perf_counter()
    res_flat = db_flat.query(test_q_vec, top_k=2)
    t_flat = (time.perf_counter() - t0) * 1000
    
    # Consulta en HNSW
    t1 = time.perf_counter()
    res_hnsw = db.query(test_q_vec, top_k=2)
    t_hnsw = (time.perf_counter() - t1) * 1000
    
    print(f"Consulta comparativa: '{test_q_text}'")
    print(f"Flat (exacto)  | Tiempo: {t_flat:.4f} ms | Vecino mas cercano: ID {res_flat[0]['id']} (Dist: {res_flat[0]['distance']:.4f})")
    print(f"HNSW (aprox.)  | Tiempo: {t_hnsw:.4f} ms | Vecino mas cercano: ID {res_hnsw[0]['id']} (Dist: {res_hnsw[0]['distance']:.4f})")

    # 4. Pruebas de Persistencia
    print("\n" + "="*50)
    print(" Serializacion e Integridad del Indice ")
    print("="*50)
    
    db_file = "data/vector_db_store.pkl"
    print(f"Guardando estado completo de la base de datos en '{db_file}'...")
    db.save(db_file)
    
    print("Cargando la base de datos desde el archivo persistido...")
    db_loaded = NanoVectorDB.load(db_file)
    
    print(f"Verificando integridad: Nodos cargados = {len(db_loaded.vectors)}")
    res_loaded = db_loaded.query(test_q_vec, top_k=1)
    print(f"Resultado en la DB cargada para '{test_q_text}': ID {res_loaded[0]['id']} | '{res_loaded[0]['metadata']['text']}'")
    
    # Limpieza del archivo creado para no saturar el espacio de desarrollo
    if os.path.exists(db_file):
        os.remove(db_file)
        # Eliminamos la carpeta temporal data si queda vacia
        try:
            os.rmdir("data")
        except OSError:
            pass


if __name__ == "__main__":
    main()
