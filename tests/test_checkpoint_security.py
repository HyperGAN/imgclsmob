"""CPU checkpoint loading must reject pickle code execution."""
import importlib.util
import os
import pickle
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    'checkpoint_model_store', Path(__file__).parents[1] / 'pytorch/pytorchcv/models/model_store.py')
store = importlib.util.module_from_spec(spec)
spec.loader.exec_module(store)


@pytest.mark.parametrize('ignore_extra', [True, False])
def test_tensor_state_dict_roundtrip(tmp_path, ignore_extra):
    expected = torch.nn.Linear(2, 1)
    checkpoint = tmp_path / 'weights.pt'
    torch.save(expected.state_dict(), checkpoint)
    restored = torch.nn.Linear(2, 1)
    store.load_model(restored, str(checkpoint), ignore_extra=ignore_extra)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[key])


@pytest.mark.parametrize('ignore_extra', [True, False])
def test_pickle_payload_rejected(tmp_path, monkeypatch, ignore_extra):
    marker = tmp_path / 'pickle-executed'

    class Payload:
        def __reduce__(self):
            return os.system, (f'touch {marker}',)

    checkpoint = tmp_path / 'hostile.pt'
    torch.save(Payload(), checkpoint)
    monkeypatch.setenv('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    with pytest.raises(pickle.UnpicklingError):
        store.load_model(torch.nn.Linear(2, 1), str(checkpoint), ignore_extra=ignore_extra)
    assert not marker.exists()
