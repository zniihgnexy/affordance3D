import torch

from affordance3d.device import ENV_VAR, default_device


def test_cuda_is_never_picked_automatically(monkeypatch) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert default_device() == "cpu"


def test_the_environment_variable_selects_the_device_explicitly(monkeypatch) -> None:
    monkeypatch.setenv(ENV_VAR, "cuda:1")

    assert default_device() == "cuda:1"


def test_apple_gpu_is_used_when_present(monkeypatch) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert default_device() == "mps"
