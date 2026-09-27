"""VSS package root.

Public API:
    from vss import VSS
    model = VSS.from_pretrained(path)
    model.decide(state=..., questions=...)
"""
from .api import VSS

__all__ = ["VSS"]
__version__ = "0.1.0"
