"""Compatibilidad: el código vive en el paquete `nano_vector_db`. Este módulo se
conserva para quien añade esta carpeta a `sys.path` e importa `database`
directamente (otros repos de ai-core-infra lo hacen)."""

from nano_vector_db.database import *  # noqa: F401,F403
from nano_vector_db.database import NanoVectorDB  # noqa: F401
