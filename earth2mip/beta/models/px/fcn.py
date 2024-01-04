# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FCN adapter from Modulus
"""
import datetime
from collections import OrderedDict
from typing import Any, Iterator

import modulus
import numpy as np
import torch

from earth2mip.beta.models.auto import AutoModel, Package
from earth2mip.beta.models.px.base import PrognosticModel
from earth2mip.beta.models.px.utils import PrognosticMixin
from earth2mip.beta.utils import handshake_coord, handshake_dim

VARIABLES = [
    "u10m",
    "v10m",
    "t2m",
    "sp",
    "msl",
    "t850",
    "u1000",
    "v1000",
    "z1000",
    "u850",
    "v850",
    "z850",
    "u500",
    "v500",
    "z500",
    "t500",
    "z50",
    "r500",
    "r850",
    "tcwv",
    "u100m",
    "v100m",
    "u250",
    "v250",
    "z250",
    "t250",
]


class FCN(torch.nn.Module, AutoModel, PrognosticMixin):
    """FourCastNet global prognostic model. Consists of a single model with a time-step
    size of 6 hours. FourCastNet operates on a [720 x 1440] equirectangular grid with 26
    variables.

    Note
    ----
    For additional information see the following resources:

    - https://arxiv.org/abs/2202.11214
    - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/modulus_fcn

    Parameters
    ----------
    core_model : torch.nn.Module
        Core PyTorch model with loaded weights
    center : torch.Tensor
        Model center normalization tensor of size [26]
    scale : torch.Tensor
        Model scale normalization tensor of size [26]
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        center: torch.Tensor,
        scale: torch.Tensor,
    ):
        super().__init__()
        self.model = core_model
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)

    input_coords = OrderedDict(
        {
            "lead_time": np.array([datetime.timedelta(0)]),
            "variable": np.array(VARIABLES),
            "lat": np.linspace(90, -90, 720, endpoint=False),
            "lon": np.linspace(90, -90, 1440, endpoint=False),
        }
    )

    output_coords = OrderedDict(
        {
            "lead_time": np.array([datetime.timedelta(hours=6)]),
            "variable": np.array(VARIABLES),
            "lat": np.linspace(90, -90, 720, endpoint=False),
            "lon": np.linspace(90, -90, 1440, endpoint=False),
        }
    )

    @classmethod
    def load_default_package(cls) -> Package:
        return Package("ngc://model/nvidia/modulus/modulus_fcnv2_sm@v0.2")

    @classmethod
    def load_model(
        cls,
        package: Package,
    ) -> PrognosticModel:
        # Load normalization vectors and core model from file system
        local_center = torch.Tensor(np.load(package.get("global_means.npy")))
        local_std = torch.Tensor(np.load(package.get("global_stds.npy")))
        core_model = modulus.Module.from_checkpoint(package.get("fcn.mdlus"))
        return cls(core_model, center=local_center, scale=local_std)

    @torch.inference_mode()
    def __call__(
        self,
        x: torch.Tensor,
        coords: OrderedDict[str, np.ndarray],
    ) -> Iterator[tuple[torch.Tensor, OrderedDict[str, np.ndarray], Any]]:
        """Create prognostic iterator

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : OrderedDict[str, np.ndarray]
            Coordinate system, should have dimensions [time, variable, lat, lon]

        Yields
        ------
        Iterator[Tuple[torch.Tensor, OrderedDict[str, np.ndarray], Any]]
            Time iterator
        """
        handshake_dim(coords, "lon", 3)
        handshake_dim(coords, "lat", 2)
        handshake_dim(coords, "variable", 1)
        handshake_dim(coords, "time", 0)
        handshake_coord(coords, self.input_coords, "lon")
        handshake_coord(coords, self.input_coords, "lat")
        handshake_coord(coords, self.input_coords, "variable")

        yield from self._default_iterator(x, coords)

    def _default_iterator(
        self, x: torch.Tensor, coords: OrderedDict[str, np.ndarray]
    ) -> Iterator[tuple[torch.Tensor, OrderedDict[str, np.ndarray], Any]]:
        coords = coords.copy()
        x = (x - self.center) / self.scale
        while True:
            # Front hook
            x, coords = self.front_hook(x, coords)
            # Forward pass
            x = self.model(x)
            coords["time"] = coords["time"] + self.output_coords["lead_time"]
            # Rear hook
            x, coords = self.rear_hook(x, coords)
            # Denormalize
            out = self.scale * x + self.center
            yield out, coords, None
