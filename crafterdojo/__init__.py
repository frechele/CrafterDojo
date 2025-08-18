try:
    from rich import traceback
    import numpy as np

    traceback.install(suppress=[np])
except ImportError:
    pass
