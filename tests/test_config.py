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
    assert dumped["envs"] == {"GUARDRAIL_API_KEY": "<set>", "PLAIN": "<set>"}  # names only, never values
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


def test_secret_key_segments():
    from src.config import is_secret_key

    for k in ("api_key", "LITELLM_API_KEY", "token", "hf_token", "password", "apiKey", "authorization", "DB_PASS"):
        assert is_secret_key(k), k
    for k in ("max_tokens", "unauthorized_advice", "top_logprobs", "api_key_env", "keywords", "env_passthrough"):
        assert not is_secret_key(k), k


def test_urls_scrubbed_envs_never_dumped(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("target: {base_url: 'https://svc:Pw0rd@litellm.internal:4000'}\ngen: {max_tokens: 64}\n"
                 "headers: {tokens: {gh: ghp_x}}\n")
    envf = tmp_path / ".env"
    envf.write_text("DATABASE_URL=postgresql://u:dbpw@db/x\nPLAIN=hello\n")
    cfg = load_config([f], env_file=envf, overrides=["envs.EXTRA_TOKEN=abc"])
    assert cfg["envs"]["EXTRA_TOKEN"] == "abc"                      # --set envs.* wins over .env
    text = dump_config(cfg, tmp_path / "o.yaml").read_text()
    for secret in ("Pw0rd", "dbpw", "hello", "ghp_x", "abc"):
        assert secret not in text, secret
    d = yaml.safe_load(text)
    assert d["gen"]["max_tokens"] == 64 and d["envs"]["PLAIN"] == "<set>"


def test_diamond_base_and_index_errors(tmp_path):
    import pytest

    (tmp_path / "d.yaml").write_text("x: 1\n")
    (tmp_path / "b.yaml").write_text("_base_: d.yaml\ny: 1\n")
    (tmp_path / "a.yaml").write_text("_base_: [d.yaml, b.yaml]\nz: 1\n")
    assert load_config([tmp_path / "a.yaml"], env_file=None)["y"] == 1
    cfg = {"l": [1, 2]}
    set_by_path(cfg, "l.2", 3)
    assert cfg["l"] == [1, 2, 3]
    with pytest.raises(ValueError, match="out of range"):
        set_by_path(cfg, "l.9", 0)
    with pytest.raises(ValueError, match="does not exist"):
        set_by_path(cfg, "missing.0", 0)


def test_referenced_file_contents_change_hash(tmp_path):
    preset = tmp_path / "preset.yaml"
    preset.write_text("threshold: 0.5\n")
    main = tmp_path / "main.yaml"
    main.write_text(f"target: {{server_config: ['{preset}']}}\n")
    h1 = load_config([main], env_file=None)["meta"]["config_hash"]
    preset.write_text("threshold: 0.7\n")
    assert load_config([main], env_file=None)["meta"]["config_hash"] != h1
