import pytest
import torch

from src.models.encoders import ModalityEncoder, ModalityEncoderBank
from src.data.synthetic_generator import MODALITY_FEATURES


def test_single_encoder_output_shape():
    enc = ModalityEncoder(in_features=500, hidden_dim=64, out_dim=128)
    x = torch.randn(100, 500)
    out = enc(x)
    assert out.shape == (100, 128)


def test_single_encoder_train_vs_eval():
    enc = ModalityEncoder(in_features=100, hidden_dim=64, out_dim=128)
    x = torch.randn(50, 100)
    enc.train()
    out_train = enc(x)
    enc.eval()
    out_eval = enc(x)
    assert out_train.shape == out_eval.shape


def test_encoder_bank_all_modalities():
    bank = ModalityEncoderBank(modality_features=MODALITY_FEATURES, out_dim=128)
    for mod, n_feat in MODALITY_FEATURES.items():
        x = torch.randn(50, n_feat)
        out = bank(mod, x)
        assert out.shape == (50, 128), f"{mod} output shape wrong"


def test_encoder_bank_unknown_modality():
    bank = ModalityEncoderBank(modality_features=MODALITY_FEATURES, out_dim=128)
    with pytest.raises(KeyError):
        bank("nonexistent", torch.randn(10, 10))
