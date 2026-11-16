import os

import pytest

pytest_plugins = ["pytester"]

_GATES = {
    "live": ("VOICE_AGENT_LIVE", "set VOICE_AGENT_LIVE=1 (paid API)"),
    "pi": ("VOICE_AGENT_PI", "set VOICE_AGENT_PI=1 (needs a Raspberry Pi)"),
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for marker, (var, reason) in _GATES.items():
        if os.environ.get(var) == "1":
            continue
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def regex_gate() -> object:
    """The changeable-question gate is the regex unless a test loads the model (`real_gate`): the
    other tests must not depend on whether out/models/eg-e5 exists on this machine."""
    from voice_agent import gate  # noqa: PLC0415

    gate.reset()
    gate._loaded.append(None)
    yield
    gate.reset()


@pytest.fixture
def real_gate() -> object:
    """The installed EverGreen gate; skips when `scripts/export_gate_onnx.sh` has not been run."""
    from voice_agent import gate  # noqa: PLC0415

    gate.reset()
    loaded = gate.load()
    if loaded is None:
        pytest.skip("run scripts/export_gate_onnx.sh to build out/models/eg-e5")
    return loaded
