import yaml

from src.config import config_hash, dump_config, load_config, parse_overrides, redact, set_by_path
from src.config.config import REDACTED


def test_override_precedence(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("a: {b: 1, c: [1, 2]}\neval: {concurrency: 4}\n")
    child = tmp_path / "child.yaml"
    child.write_text(f"_base_: {base.name}\na: {{b: 2}}\n")
    cfg = load_config([child], args={"concurrency": 8, "unused": None}, arg_to_key={"concurrency": "eval.concurrency"},
                      overrides=["a.c=[3]", "x.y=true"], env_file=None)
    assert cfg["a"] == {"b": 2, "c": [3]}
    assert cfg["eval"]["concurrency"] == 8
    assert cfg["x"]["y"] is True
    assert cfg["meta"]["cli_args"] == {"concurrency": 8}


def test_none_args_do_not_clobber_yaml(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("eval: {concurrency: 4}\n")
    cfg = load_config([f], args={"concurrency": None}, arg_to_key={"concurrency": "eval.concurrency"}, env_file=None)
    assert cfg["eval"]["concurrency"] == 4


def test_env_loaded_and_redacted_on_dump(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("server: {api_key_env: GUARDRAIL_API_KEY}\n")
    envf = tmp_path / ".env"
    envf.write_text("GUARDRAIL_API_KEY=super-secret\nPLAIN=value\n")
    cfg = load_config([f], env_file=envf)
    assert cfg["envs"]["GUARDRAIL_API_KEY"] == "super-secret"
    out = dump_config(cfg, tmp_path / "out" / "config.yaml")
    dumped = yaml.safe_load(out.read_text())
    assert dumped["envs"]["GUARDRAIL_API_KEY"] == REDACTED
    assert dumped["envs"]["PLAIN"] == "value"
    assert dumped["server"]["api_key_env"] == "GUARDRAIL_API_KEY"  # *_env holds a name, not a secret
    assert "super-secret" not in out.read_text()


def test_config_hash_ignores_meta_and_envs():
    a = {"x": 1, "meta": {"t": 1}, "envs": {"K": "1"}}
    b = {"x": 1, "meta": {"t": 2}, "envs": {"K": "2"}}
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash({"x": 2})


def test_list_index_override():
    cfg = {"pipeline": {"stages": [{"threshold": 0.5}, {"threshold": 0.5}]}}
    set_by_path(cfg, "pipeline.stages.1.threshold", 0.7)
    assert cfg["pipeline"]["stages"][1]["threshold"] == 0.7
    assert parse_overrides(["k=a=b"]) == [("k", "a=b")]
    assert redact({"api_key": "x", "token_env": "T"}) == {"api_key": REDACTED, "token_env": "T"}
