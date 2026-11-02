"""
Pipeline Package for S.W.O.R.D. v2 Persistent Video Tracking.
"""

def run_pipeline(*args, **kwargs):
    from .run_generic_pipeline import run_pipeline as _run_pipeline
    return _run_pipeline(*args, **kwargs)

def load_config(*args, **kwargs):
    from .run_generic_pipeline import load_config as _load_config
    return _load_config(*args, **kwargs)

def validate_run(*args, **kwargs):
    from .validate_pipeline import validate_run as _validate_run
    return _validate_run(*args, **kwargs)

def __getattr__(name):
    if name in ("PersistentTrackingValidator", "Phase2Validator"):
        from .validate_pipeline import PersistentTrackingValidator
        return PersistentTrackingValidator
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["run_pipeline", "load_config", "validate_run", "PersistentTrackingValidator", "Phase2Validator"]


