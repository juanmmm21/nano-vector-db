"""Compatibilidad: el código vive en el paquete `nano_vector_db`. Este módulo se
conserva para quien añade esta carpeta a `sys.path` e importa `distance`
directamente (otros repos de ai-core-infra lo hacen)."""

from nano_vector_db.distance import *  # noqa: F401,F403
