from __future__ import annotations

from pathlib import Path

import sanitize_trace


def test_site_env_override_survives_a_frozen_source_without_dotenv(
    tmp_path: Path, monkeypatch
) -> None:
    frozen_source = tmp_path / "frozen-source"
    frozen_source.mkdir()
    site_env = tmp_path / "site/runtime.env"
    site_env.parent.mkdir()
    site_env.write_text(
        'OPENAI_API_KEY="site-secret-value"\nPLACEHOLDER_API_KEY="your-value"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(sanitize_trace, "DEFAULT_ENV_PATH", frozen_source / ".env")
    monkeypatch.setenv("POST_TRAIN_BENCH_ENV_FILE", str(site_env))

    assert sanitize_trace.configured_env_path() == site_env
    assert sanitize_trace.load_api_key_secrets() == {
        "OPENAI_API_KEY": "site-secret-value"
    }


def test_explicit_env_path_still_overrides_the_site_environment(
    tmp_path: Path, monkeypatch
) -> None:
    site_env = tmp_path / "site.env"
    explicit = tmp_path / "explicit.env"
    site_env.write_text('OPENAI_API_KEY="site-secret-value"\n', encoding="utf-8")
    explicit.write_text('OPENAI_API_KEY="explicit-secret-value"\n', encoding="utf-8")
    monkeypatch.setenv("POST_TRAIN_BENCH_ENV_FILE", str(site_env))

    assert sanitize_trace.load_api_key_secrets(explicit) == {
        "OPENAI_API_KEY": "explicit-secret-value"
    }
