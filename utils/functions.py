"""Dynamic import helpers for configured model and evaluator classes."""

import importlib


__all__ = ["load_model_class"]


def load_model_class(identifier: str, prefix: str = "models."):
    module_path, class_name = identifier.split('@')

    # Import the module
    module = importlib.import_module(prefix + module_path)
    cls = getattr(module, class_name)
    
    return cls
