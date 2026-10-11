import pytest
from pydantic import ValidationError

from hybrid_router.backends.base import Backend, BackendConfig
from hybrid_router.backends.ollama import OllamaConfig
from hybrid_router.backends.openai_compat import OpenAICompatConfig
from hybrid_router.backends.registry import create_backend, register_backend
from hybrid_router.config import env_overrides, load_config
from hybrid_router.types import Tier


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HYBRID_ROUTER_CONFIG", raising=False)


def test_defaults_without_a_file():
    cfg = load_config(environ={})
    assert isinstance(cfg.tiers[Tier.LOCAL], OllamaConfig)
    assert cfg.tiers[Tier.LOCAL].model == "qwen3:8b"
    sonnet, opus = cfg.tiers[Tier.SONNET], cfg.tiers[Tier.OPUS]
    assert isinstance(sonnet, OpenAICompatConfig) and isinstance(opus, OpenAICompatConfig)
    assert sonnet.model == "anthropic/claude-sonnet-5.5"
    assert opus.model == "anthropic/claude-opus-5.5"
    assert sonnet.base_url == "https://openrouter.ai/api/v1"
    # Claude 5.5 rejects these, so the Claude tiers ship with them off.
    assert not sonnet.sampling_params and not opus.forced_tool_choice


def test_file_merges_over_defaults(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text("tiers:\n  local:\n    model: llama3.1:8b\n    num_ctx: 8192\n")
    cfg = load_config(path, environ={})
    local = cfg.tiers[Tier.LOCAL]
    assert isinstance(local, OllamaConfig)
    assert (local.model, local.num_ctx, local.base_url) == (
        "llama3.1:8b",
        8192,
        "http://localhost:11434",
    )


def test_default_file_in_cwd_is_picked_up(tmp_path):
    (tmp_path / "hybrid_router.yaml").write_text(
        "tiers:\n  sonnet:\n    reasoning: {effort: medium}\n"
    )
    sonnet = load_config(environ={}).tiers[Tier.SONNET]
    assert isinstance(sonnet, OpenAICompatConfig)
    assert sonnet.reasoning == {"effort": "medium"}
    assert not sonnet.sampling_params  # defaults for the tier are kept


def test_env_overrides_are_typed_and_win(tmp_path):
    (tmp_path / "hybrid_router.yaml").write_text("tiers:\n  local:\n    num_ctx: 8192\n")
    env = {
        "HYBRID_ROUTER__TIERS__LOCAL__NUM_CTX": "4096",
        "HYBRID_ROUTER__TIERS__LOCAL__THINK": "true",
        "UNRELATED": "x",
    }
    local = load_config(environ=env).tiers[Tier.LOCAL]
    assert isinstance(local, OllamaConfig)
    assert local.num_ctx == 4096 and local.think is True
    assert env_overrides({"UNRELATED": "1"}) == {}


def test_unknown_field_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("tiers:\n  local:\n    modle: typo\n")
    with pytest.raises(ValidationError, match="modle"):
        load_config(path, environ={})


def test_unknown_backend_kind_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("tiers:\n  local:\n    kind: nope\n    model: x\n")
    with pytest.raises(ValidationError, match="unknown backend kind 'nope'"):
        load_config(path, environ={})


def test_missing_file_is_an_error():
    with pytest.raises(FileNotFoundError):
        load_config("does-not-exist.yaml", environ={})


def test_third_party_backend_plugs_in_without_router_changes(tmp_path):
    class MlxConfig(BackendConfig):
        kind: str = "mlx-test"
        quantization: str = "4bit"

    @register_backend
    class MlxBackend(Backend):
        kind = "mlx-test"
        config_model = MlxConfig

        @property
        def capabilities(self):  # pragma: no cover - not exercised
            raise NotImplementedError

        def stream(self, req, *, logprobs=False):  # pragma: no cover
            raise NotImplementedError

        async def health(self):  # pragma: no cover
            raise NotImplementedError

    path = tmp_path / "mlx.yaml"
    path.write_text("tiers:\n  local:\n    kind: mlx-test\n    model: qwen3-8b-mlx\n")
    cfg = load_config(path, environ={})
    local = cfg.tiers[Tier.LOCAL]
    # Switching kind replaces the tier block, so no Ollama fields leak in.
    assert isinstance(local, MlxConfig) and local.quantization == "4bit"
    backend = create_backend(local, tier=Tier.LOCAL)
    assert isinstance(backend, MlxBackend) and backend.name == "local:mlx-test"
    dumped = cfg.model_dump(mode="json")
    assert dumped["tiers"]["local"]["quantization"] == "4bit"
