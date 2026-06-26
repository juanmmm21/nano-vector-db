# nano-vector-db

Base de datos vectorial ligera en memoria implementada desde cero en Python. Este submódulo proporciona almacenamiento indexado eficiente para busquedas rapidas de similitud de vectores de alta dimension utilizando busqueda exacta (Flat) y busqueda aproximada de vecinos mas cercanos (ANN) mediante el algoritmo HNSW (Hierarchical Navigable Small World).

## Arquitectura y Componentes Tecnicos

El motor de la base de datos se basa en dos esquemas de indexacion y recuperacion:

### 1. Búsqueda Exacta (Flat Index)
Realiza una busqueda secuencial por fuerza bruta comparando el vector consulta con todos los vectores registrados en la base de datos.
*   **Complejidad:** O(N), donde N es el numero total de vectores.
*   **Ventaja:** Exactitud del 100% y soporte de pre-filtrado nativo eficiente sobre metadatos.

### 2. Búsqueda Aproximada (HNSW Index)
Implementa el algoritmo de grafos jerarquicos multicapa para aproximacion rapida. Estructura los vectores en niveles o capas:
*   **Capa Superior:** Grafos dispersos con enlaces largos para saltos rapidos de gran escala (optimizacion de exploracion).
*   **Capa Inferior (Capa 0):** Contiene la totalidad de los vectores con enlaces densos y de corto alcance para precision local.
*   **Complejidad:** O(log N) para busqueda e insercion, lo que permite escalar a millones de vectores.

Hiperparametros de control del grafo:
*   `M`: Cantidad maxima de enlaces bidireccionales por nodo en cada capa > 0.
*   `M0`: Cantidad maxima de enlaces por nodo en la Capa 0 (fijado en `2 * M`).
*   `efConstruction`: Numero de vecinos candidatos evaluados durante la insercion. Controla el equilibrio entre precision del grafo y tiempo de construccion.
*   `efSearch`: Numero de candidatos dinamicos evaluados en la busqueda. Valores mas altos incrementan el Recall (precision de vecinos) a cambio de mayor tiempo de computo.

## Fundamentos Matematicos de Distancia

La base de datos admite tres metricas optimizadas mediante NumPy para evaluar la cercania semantica de los vectores:

### Distancia de Coseno
Mide la diferencia angular entre dos vectores, ignorando su magnitud:

$$D_{cos}(u, v) = 1.0 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$

### Distancia L2 (Euclidea)
Mide la distancia fisica en linea recta en el espacio cartesiano multi-dimensional:

$$D_{L2}(u, v) = \sqrt{\sum_{i=1}^{d} (u_i - v_i)^2}$$

### Producto Escalar Invertido
Adecuado si los vectores ya estan normalizados L2, donde el producto escalar es directamente proporcional a la similitud de coseno:

$$D_{dot}(u, v) = - (u \cdot v)$$

## Filtrado de Metadatos

Admite pre-filtrado (en busquedas Flat) y post-filtrado (en busquedas HNSW) evaluando condiciones estructuradas compatibles con los operadores tradicionales de MongoDB:
*   `$eq`: Igualdad estricta de valores.
*   `$ne`: Desigualdad o exclusion de valores.
*   `$gt` / `$gte`: Mayor que / Mayor o igual que para campos numericos.
*   `$lt` / `$lte`: Menor que / Menor o igual que.
*   `$in`: Pertenencia a una lista de elementos validos.
*   `$nin`: No pertenencia a una lista de elementos.

*Fallback Inteligente:* Si un filtro es extremadamente restrictivo y la busqueda HNSW con post-filtrado no consigue llenar el cupo de resultados `top_k`, el motor realiza un fallback automatico y transparente a la busqueda exacta `Flat` con pre-filtrado sobre los datos para asegurar el retorno de los vecinos mas cercanos existentes.

## Conexion con el Ecosistema

Este proyecto aprovecha la salida de otros modulos:
*   **contrastive-embedding-trainer:** El script `example.py` busca automaticamente el directorio de salida de este proyecto en busca de pesos y tokenizadores. De encontrarlos, los carga en PyTorch para generar embeddings semanticos reales a partir de texto plano en lugar de depender de simulaciones sinteticas.

## Instalacion y Uso

### 1. Preparar el Entorno
Crea y activa un entorno virtual en la carpeta del proyecto e instala la dependencia NumPy:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ejecutar Pruebas Automatizadas
Para verificar la integridad del grafo, calculo de distancias, filtros y serializacion:
```bash
python -m unittest test_db.py
```

### 3. Ejecutar Demostración
Para indexar documentos multitopicos reales (o sinteticos si no se ha entrenado el modelo previo) y contrastar la velocidad/precision entre la busqueda Flat y HNSW:
```bash
python example.py
```
El script demostrara la insercion, consulta estructurada con filtros, la comparativa de velocidad de ejecucion en milisegundos y la persistencia en disco de la base de datos vectorial.
