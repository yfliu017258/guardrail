"""litellm_admin against a fake in-memory proxy that mimics the 1.92.0 admin endpoints."""

import json
import uuid

import httpx
import pytest

from src.litellm_admin.__main__ import main


class FakeProxy:
    def __init__(self):
        self.db: dict[str, dict] = {}
        self.teams = {"team-1": {"guardrails": ["other-guard"]}}
        self.calls: list[tuple[str, str]] = []

    def add(self, name, managed_by=None, location="db", **lp):
        gid = str(uuid.uuid4())
        self.db[gid] = {"guardrail_id": gid, "guardrail_name": name, "guardrail_definition_location": location,
                        "litellm_params": {"guardrail": "generic_guardrail_api", "mode": "pre_call", "default_on": False,
                                           "api_key": "sk-****", **lp},
                        "guardrail_info": {"managed_by": managed_by} if managed_by else {}}
        return gid

    def handler(self, req: httpx.Request) -> httpx.Response:
        assert req.headers["authorization"] == "Bearer admin-key"
        path, method = req.url.path, req.method
        self.calls.append((method, path))
        body = json.loads(req.content) if req.content else {}
        if path == "/v2/guardrails/list":
            return httpx.Response(200, json={"guardrails": list(self.db.values())})
        if path == "/guardrails" and method == "POST":
            g = body["guardrail"]
            gid = str(uuid.uuid4())
            self.db[gid] = {**g, "guardrail_id": gid, "litellm_params": {**g["litellm_params"], "api_key": "sk-****",
                                                                          "unreachable_fallback": g["litellm_params"].get("unreachable_fallback", "fail_closed")}}
            return httpx.Response(200, json=self.db[gid])
        if path.startswith("/guardrails/") and method == "PUT":
            gid = path.rsplit("/", 1)[1]
            self.db[gid] = {**body["guardrail"], "guardrail_id": gid}
            return httpx.Response(200, json=self.db[gid])
        if path.startswith("/guardrails/") and method == "DELETE":
            self.db.pop(path.rsplit("/", 1)[1])
            return httpx.Response(200, json={})
        if path == "/team/info":
            tid = req.url.params["team_id"]
            return httpx.Response(200, json={"team_id": tid, "team_info": {"metadata": self.teams[tid]}})
        if path == "/team/update":
            self.teams[body["team_id"]] = {"guardrails": body["guardrails"]}
            return httpx.Response(200, json={})
        if path == "/guardrails/apply_guardrail":
            if "忽略" in body["text"]:
                return httpx.Response(400, json={"error": {"message": "Guardrail blocked"}})
            return httpx.Response(200, json={"response_text": body["text"]})
        return httpx.Response(404, text=f"no route {method} {path}")


@pytest.fixture()
def env(tmp_path, monkeypatch):
    e = tmp_path / ".env"
    e.write_text("LITELLM_BASE_URL=http://proxy\nLITELLM_ADMIN_KEY=admin-key\nOJT_GUARDRAIL_API_BASE=http://svc:8080\n"
                 "OJT_GUARDRAIL_CANDIDATE_API_BASE=http://svc2:8080\nOJT_GUARDRAIL_API_KEY=secret-xyz\n")
    return str(e)


def run(fake, env, *args):
    return main(["--env-file", env, "--no-audit", *args], transport=httpx.MockTransport(fake.handler))


def test_apply_dry_run_then_create_then_noop(env, capsys):
    fake = FakeProxy()
    assert run(fake, env, "apply") == 0 and not fake.db                     # dry run: nothing written
    assert run(fake, env, "apply", "--yes") == 0 and len(fake.db) == 3
    sent = [g for g in fake.db.values() if g["guardrail_name"] == "ojt-guard-pre"][0]
    assert sent["litellm_params"]["api_base"] == "http://svc:8080" and sent["guardrail_info"]["managed_by"] == "ojt-guardrail"
    capsys.readouterr()
    assert run(fake, env, "plan") == 0
    assert "noop=3" in capsys.readouterr().out                               # masked api_key is not a diff


def test_update_only_changed_and_refuse_foreign(env, capsys):
    fake = FakeProxy()
    run(fake, env, "apply", "--yes")
    gid = next(k for k, g in fake.db.items() if g["guardrail_name"] == "ojt-guard-post")
    fake.db[gid]["litellm_params"]["streaming_sampling_rate"] = 10           # drift
    fake.add("ojt-guard-pre-candidate-old", managed_by="ojt-guardrail")      # stale managed one
    fake.add("someone-elses", managed_by="security-team")
    capsys.readouterr()
    run(fake, env, "apply", "--yes", "--prune")
    out = capsys.readouterr().out
    assert "update   ojt-guard-post" in out and "streaming_sampling_rate: 10 -> 5" in out
    names = {g["guardrail_name"] for g in fake.db.values()}
    assert "ojt-guard-pre-candidate-old" not in names and "someone-elses" in names
    puts = [c for c in fake.calls if c[0] == "PUT"]
    assert len(puts) == 1


def test_conflict_with_foreign_or_config_guardrail(env, capsys):
    fake = FakeProxy()
    fake.add("ojt-guard-pre", managed_by="security-team")
    fake.add("ojt-guard-post", location="config")
    assert run(fake, env, "apply", "--yes") == 1          # conflicts -> non-zero exit, others still applied
    out = capsys.readouterr().out
    assert out.count("conflict") >= 2 and "create   ojt-guard-pre-candidate" in out
    assert [c for c in fake.calls if c[0] == "PUT"] == []


def test_attach_add_and_test(env, capsys):
    fake = FakeProxy()
    run(fake, env, "apply", "--yes")
    run(fake, env, "attach", "--team", "team-1", "--guardrails", "ojt-guard-pre", "--yes")
    assert fake.teams["team-1"]["guardrails"] == ["other-guard", "ojt-guard-pre"]
    run(fake, env, "attach", "--team", "team-1", "--guardrails", "other-guard", "--mode", "remove", "--yes")
    assert fake.teams["team-1"]["guardrails"] == ["ojt-guard-pre"]
    capsys.readouterr()
    run(fake, env, "test", "ojt-guard-pre")
    out = capsys.readouterr().out
    assert "[PASS] 今天天氣如何？" in out and "BLOCKED/ERROR HTTP 400" in out


def test_missing_env_ref_fails(tmp_path):
    e = tmp_path / ".env"
    e.write_text("LITELLM_BASE_URL=http://proxy\nLITELLM_ADMIN_KEY=admin-key\n")
    with pytest.raises(SystemExit):
        main(["--env-file", str(e), "--no-audit", "plan"], transport=httpx.MockTransport(FakeProxy().handler))
