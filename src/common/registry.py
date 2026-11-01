"""One generic plug-in registry used by every extensible part of the project.

    CONVERTERS      src/converters/base.py       dataset -> GuardrailSample
    PARSERS         src/server/backends/parsers.py  guard-model output -> verdict
    BACKENDS        src/server/backends/base.py  guard stage implementations
    PROMPT_BUILDERS src/server/backends/prompts.py  how a guard model is prompted
    TARGETS         src/evaluation/targets.py    what a guardrail eval talks to
    TASKS           src/evaluation/tasks/base.py guardrail / refusal_judge / model_safety
    GENERATORS      src/evaluation/generation.py bare-model clients
    REFUSAL_JUDGES  src/refuse_judge/base.py     refusal detection strategies

Usage::

    BACKENDS = Registry("guard backend", base=GuardBackend, lazy_modules=["src.server.backends.regex", ...])

    @BACKENDS.register("regex")
    class RegexBackend(GuardBackend): ...

    backend = BACKENDS.build({"type": "regex", ...}, root_cfg)   # -> cls.from_config(cfg, root_cfg)

``lazy_modules`` are imported on first lookup, so a config can name any registered type without
the caller importing its module (and without circular imports at package import time).
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Generic, Iterable, Mapping, TypeVar

T = TypeVar("T")


class RegistryError(KeyError):
    def __str__(self) -> str:  # KeyError repr-quotes its message; keep it readable
        return str(self.args[0])


class Registry(Generic[T]):
    def __init__(self, kind: str, base: type | None = None, lazy_modules: Iterable[str] = ()):
        self.kind = kind
        self.base = base
        self._lazy = list(lazy_modules)
        self._loaded = False
        self._items: dict[str, type[T]] = {}
        self._aliases: dict[str, str] = {}
        self._import_errors: dict[str, str] = {}

    # -- registration ----------------------------------------------------- #
    def register(self, name: str | None = None, *, aliases: Iterable[str] = ()) -> Callable[[type[T]], type[T]]:
        def deco(cls: type[T]) -> type[T]:
            key = name or getattr(cls, "registry_name", None) or cls.__name__
            if self.base is not None and not issubclass(cls, self.base):
                raise TypeError(f"{cls.__name__} must subclass {self.base.__name__} to be a {self.kind}")
            existing = self._items.get(key)
            if existing is not None and existing is not cls and existing.__qualname__ != cls.__qualname__:
                raise RegistryError(f"duplicate {self.kind} name {key!r}: {existing.__qualname__} vs {cls.__qualname__}")
            self._items[key] = cls
            cls.registry_name = key  # type: ignore[attr-defined]
            for a in aliases:
                self._aliases[a] = key
            return cls

        return deco

    def add_lazy_module(self, module: str) -> None:
        self._lazy.append(module)
        self._loaded = False

    # -- lookup ----------------------------------------------------------- #
    def _load(self) -> None:
        """Import plug-in modules once. A broken module must not hide the others (e.g. an optional
        dependency missing for one backend); its error is reported only when a lookup fails."""
        if not self._loaded:
            self._loaded = True
            for m in self._lazy:
                try:
                    importlib.import_module(m)
                except Exception as e:  # noqa: BLE001 - surfaced in get()
                    self._import_errors[m] = f"{type(e).__name__}: {e}"

    def get(self, name: str) -> type[T]:
        self._load()
        key = self._aliases.get(name, name)
        if key not in self._items:
            msg = f"unknown {self.kind} {name!r}; available: {', '.join(self.names())}"
            if self._import_errors:
                msg += "; modules that failed to import: " + "; ".join(f"{m} ({e})" for m, e in self._import_errors.items())
            raise RegistryError(msg)
        return self._items[key]

    def build(self, cfg: Mapping[str, Any], *args: Any, type_key: str = "type", default: str | None = None,
              **kwargs: Any) -> T:
        """Instantiate from a config dict: ``cfg[type_key]`` selects the class, which is built with
        ``cls.from_config(cfg, *args, **kwargs)`` when defined, else ``cls(cfg, *args, **kwargs)``."""
        name = cfg.get(type_key, default)
        if name is None:
            raise RegistryError(f"{self.kind} config needs a {type_key!r} key; available: {', '.join(self.names())}")
        cls = self.get(name)
        factory = getattr(cls, "from_config", None)
        return factory(cfg, *args, **kwargs) if callable(factory) else cls(cfg, *args, **kwargs)  # type: ignore[call-arg]

    @property
    def import_errors(self) -> dict[str, str]:
        """Plug-in modules that failed to import (module -> error). CLIs should print these."""
        self._load()
        return dict(self._import_errors)

    def names(self) -> list[str]:
        self._load()
        return sorted(self._items)

    def items(self) -> list[tuple[str, type[T]]]:
        self._load()
        return sorted(self._items.items())

    def describe(self) -> list[dict[str, str]]:
        """``[{name, class, doc}]`` — used by ``--list`` CLIs."""
        out = []
        for name, cls in self.items():
            doc = (cls.__doc__ or "").strip().splitlines()
            out.append({"name": name, "class": f"{cls.__module__}.{cls.__qualname__}", "doc": doc[0] if doc else ""})
        return out

    def __contains__(self, name: object) -> bool:
        self._load()
        return name in self._items or name in self._aliases

    def __len__(self) -> int:
        self._load()
        return len(self._items)

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {len(self._items)} items)"
