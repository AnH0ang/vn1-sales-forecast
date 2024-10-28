import torch
from neuralforecast.losses.pytorch import BasePointLoss


class CustomLoss(BasePointLoss):
    def __init__(self, horizon_weight=None) -> None:
        super().__init__(
            horizon_weight=horizon_weight,
            outputsize_multiplier=1,
            output_names=[""],
        )

    def __call__(
        self, y: torch.Tensor, y_hat: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        mae = torch.abs(y - y_hat).sum()
        bias = torch.abs((y - y_hat).sum())
        score = mae + bias

        score = score / y.abs().sum()

        score[score != score] = 0.0  # noqa: PLR0124
        score[score == float("inf")] = 0.0
        return score
