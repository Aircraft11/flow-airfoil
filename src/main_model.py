"""Load a reversible flow matching checkpoint and its normalization metadata."""

import torch

from .models import BidirectionalAirfoilFlow

DEFAULT_TEMPERATURE = 0.35


def load_model(checkpoint, device):
    """Rebuild the inference model and return it with its checkpoint metadata."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if len(saved["shape_columns"]) != 20 or len(saved["aero_columns"]) != 6:
        raise ValueError("Expected a reversible flow matching checkpoint with 20 CST coefficients")
    model = BidirectionalAirfoilFlow(saved["config"]["model"])
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval(), saved
