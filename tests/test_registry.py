import pytest

from src.common import Registry, RegistryError


class Base:
    def __init__(self, cfg, extra=None):
        self.cfg, self.extra = cfg, extra


def test_register_build_and_errors():
    reg = Registry("thing", base=Base)

    @reg.register("a", aliases=["alpha"])
    class A(Base):
        """A thing."""

    @reg.register("b")
    class B(Base):
        @classmethod
        def from_config(cls, cfg, extra=None):
            obj = cls(cfg, extra)
            obj.via_factory = True
            return obj

    assert reg.get("alpha") is A and A.registry_name == "a" and "alpha" in reg
    assert reg.build({"type": "a", "x": 1}, "e").extra == "e"
    assert reg.build({"type": "b"}).via_factory
    assert reg.describe()[0] == {"name": "a", "class": f"{A.__module__}.{A.__qualname__}", "doc": "A thing."}
    with pytest.raises(RegistryError, match="available: a, b"):
        reg.get("c")
    with pytest.raises(RegistryError, match="needs a 'type'"):
        reg.build({})
    with pytest.raises(TypeError):
        reg.register("bad")(type("NotBase", (), {}))
    with pytest.raises(RegistryError, match="duplicate"):
        reg.register("a")(type("Other", (Base,), {}))


def test_lazy_module_failure_is_reported_not_fatal():
    reg = Registry("thing", lazy_modules=["src.does_not_exist"])

    @reg.register("ok")
    class Ok:
        pass

    assert reg.get("ok") is Ok                     # a broken plug-in module does not hide registered ones
    with pytest.raises(RegistryError, match="does_not_exist"):
        reg.get("missing")
