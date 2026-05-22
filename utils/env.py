import os
from typing import Optional, Tuple


__all__ = [
    "env_name_candidates",
    "get_env",
    "truthy_env",
]


_STRICT_TRUTHY = {"1", "true", "yes", "y", "on"}


def env_name_candidates(name: str, *, legacy_name: Optional[str] = None) -> Tuple[str, ...]:
    raw_name = str(name).strip()
    if not raw_name:
        return tuple()

    names = []

    def _add(value: Optional[str]) -> None:
        if not value:
            return
        normalized = str(value).strip()
        if normalized and normalized not in names:
            names.append(normalized)

    _add(raw_name)
    if legacy_name is not None:
        _add(legacy_name)
    elif raw_name.startswith("HRM_"):
        _add(raw_name[4:])
    else:
        _add(f"HRM_{raw_name}")

    return tuple(names)


def get_env(name: str, default: Optional[str] = None, *, legacy_name: Optional[str] = None) -> Optional[str]:
    for candidate in env_name_candidates(name, legacy_name=legacy_name):
        value = os.environ.get(candidate)
        if value is not None:
            return value
    return default


def truthy_env(name: str, *, legacy_name: Optional[str] = None) -> bool:
    value = get_env(name, legacy_name=legacy_name)
    if value is None:
        return False
    return value.strip().lower() in _STRICT_TRUTHY
