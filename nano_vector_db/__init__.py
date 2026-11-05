"""Vector database from scratch: exact (flat) and HNSW approximate nearest-neighbour search with metadata filters."""

from .database import NanoVectorDB
from .hnsw import HNSWIndex

__version__ = "0.1.0"

__all__ = ['NanoVectorDB', 'HNSWIndex']
