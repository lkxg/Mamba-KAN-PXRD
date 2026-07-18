from __future__ import annotations

import unittest

import torch

from src.models import XRDCTMClassifier
from src.training import loss_with_contrastive


def build_small_model() -> XRDCTMClassifier:
    return XRDCTMClassifier(
        in_dim=128,
        num_classes=230,
        stem_channels=16,
        branch_dropout=0.0,
        cnn={
            "stem_factor": 4,
            "mid_channels": 24,
            "feature_channels": 32,
            "stage1_blocks": 1,
            "stage2_blocks": 1,
            "stage3_blocks": 1,
            "dropout": 0.0,
            "pool_hidden": 32,
        },
        mamba={
            "backend": "auto",
            "bidirectional": False,
            "mid_channels": 24,
            "stage1_layers": 0,
            "stage2_layers": 0,
            "d_state": 8,
            "d_conv": 4,
            "expand": 2,
            "headdim": None,
            "chunk_size": None,
            "dropout": 0.0,
            "pooling_dropout": 0.0,
        },
        peak={
            "num_peaks": 8,
            "detector_window": 5,
            "prominence_window": 9,
            "min_prominence": 0.01,
            "d_model": 32,
            "num_layers": 1,
            "num_heads": 4,
            "ffn_dim": 64,
            "dropout": 0.0,
            "feedback_sigma": 1.5,
        },
        fusion={"feedback_scale": 0.1, "dropout": 0.0},
    )


class XRDCTMClassifierTest(unittest.TestCase):
    def test_forward_contract(self) -> None:
        model = build_small_model().eval()
        x = torch.zeros(2, 128)
        x[:, 10] = 1.0
        x[:, 70] = 0.6
        with torch.no_grad():
            output = model(x)

        self.assertEqual(tuple(output["logits"].shape), (2, 230))
        self.assertEqual(set(output["aux_logits"]), {"cnn", "mamba", "peak"})
        for logits in output["aux_logits"].values():
            self.assertEqual(tuple(logits.shape), (2, 230))
            self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(tuple(output["gate_weights"].shape), (2, 3))
        self.assertTrue(
            torch.allclose(
                output["gate_weights"].sum(dim=-1),
                torch.ones(2),
                atol=1.0e-6,
            )
        )
        self.assertEqual(tuple(output["quality"].shape), (2, 4))

    def test_auxiliary_loss_reaches_all_branches(self) -> None:
        model = build_small_model().train()
        x = torch.rand(2, 128)
        target = torch.tensor([0, 229])
        output = model(x)
        loss, _, _ = loss_with_contrastive(
            output,
            target,
            torch.nn.CrossEntropyLoss(),
            auxiliary_weight=0.15,
        )
        loss.backward()

        gradients = [
            model.stem.proj.weight.grad,
            model.cnn_branch.down1.conv.weight.grad,
            model.mamba_branch.down1.conv.weight.grad,
            model.peak_branch.peak_embed[0].weight.grad,
            model.fusion_gate[1].weight.grad,
        ]
        self.assertTrue(
            all(
                grad is not None
                and torch.isfinite(grad).all()
                and float(grad.abs().sum()) > 0.0
                for grad in gradients
            )
        )

    def test_bidirectional_mamba_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unidirectional"):
            XRDCTMClassifier(
                in_dim=128,
                num_classes=230,
                mamba={"bidirectional": True},
            )


if __name__ == "__main__":
    unittest.main()
