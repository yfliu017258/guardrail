"""src.server.check against a real server."""

from src.server.check import BUILTIN_CASES, Case, main, run_checks
from src.server.testing import BackgroundServer


async def test_check_passes_on_the_mock_server(mock_server):
    rep = await run_checks(mock_server.url, None, BUILTIN_CASES, load=20, concurrency=4)
    fails = [r for r in rep.rows if r[0] == "FAIL"]
    assert not fails, fails
    assert ("PASS", "ready") == rep.rows[2][:2]
    assert ("WARN", "auth") == rep.rows[3][:2]          # mock.yaml runs without auth -> warned


async def test_check_reports_a_wrong_decision(mock_server):
    rep = await run_checks(mock_server.url, None, [Case("must_block", "BLOCKED", ["今天天氣很好"])], load=0, concurrency=1)
    assert rep.failed and any(r[1] == "case:must_block" and r[0] == "FAIL" for r in rep.rows)


def test_cli_exit_codes(mock_server, tmp_path):
    assert main(["--url", mock_server.url, "--env-file", str(tmp_path / "none")]) == 0
    cases = tmp_path / "cases.yaml"
    cases.write_text("- {name: benign_zh_tw, texts: ['今天天氣很好'], expect: BLOCKED}\n")
    assert main(["--url", mock_server.url, "--cases", str(cases), "--env-file", str(tmp_path / "none")]) == 1
    assert main(["--url", "http://127.0.0.1:9", "--env-file", str(tmp_path / "none")]) == 1


def test_auth_is_checked(tmp_path):
    with BackgroundServer(["src/config/server/mock.yaml"], overrides=["server.api_key_env=GUARDRAIL_API_KEY"],
                          envs={"GUARDRAIL_API_KEY": "k"}) as srv:
        envf = tmp_path / ".env"
        envf.write_text("GUARDRAIL_API_KEY=k\n")
        assert main(["--url", srv.url, "--env-file", str(envf)]) == 0
