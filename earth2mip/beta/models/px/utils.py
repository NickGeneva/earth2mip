from collections import OrderedDict

import numpy as np
import torch

from earth2mip.beta.models.utils import AutoModel


class PrognosticMixin(AutoModel):
    def _default_hook(
        self, x: torch.Tensor, coords: OrderedDict[str, np.ndarray]
    ) -> tuple[torch.Tensor, OrderedDict[str, np.ndarray]]:
        return x, coords

    front_hook = _default_hook
    rear_hook = _default_hook
