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
FCN v2 Small adapter

This model is an outdated version of FCN v2 (SFNO), a more recent one is present in Modulus.
"""
import datetime
import pathlib
from collections import OrderedDict
from typing import Any, Iterator

import numpy as np
import torch

# TODO: Update to new arch in Modulus!
import earth2mip.beta.models.nn.fcnv2 as fcnv2
from earth2mip.beta.models.auto import AutoModel, Package
from earth2mip.beta.models.px.base import PrognosticModel
from earth2mip.beta.models.px.utils import PrognosticMixin
from earth2mip.beta.utils import handshake_coord, handshake_dim

VARIABLES = [
    "u10m",
    "v10m",
    "u100m",
    "v100m",
    "t2m",
    "sp",
    "msl",
    "tcwv",
    "u50",
    "u100",
    "u150",
    "u200",
    "u250",
    "u300",
    "u400",
    "u500",
    "u600",
    "u700",
    "u850",
    "u925",
    "u1000",
    "v50",
    "v100",
    "v150",
    "v200",
    "v250",
    "v300",
    "v400",
    "v500",
    "v600",
    "v700",
    "v850",
    "v925",
    "v1000",
    "z50",
    "z100",
    "z150",
    "z200",
    "z250",
    "z300",
    "z400",
    "z500",
    "z600",
    "z700",
    "z850",
    "z925",
    "z1000",
    "t50",
    "t100",
    "t150",
    "t200",
    "t250",
    "t300",
    "t400",
    "t500",
    "t600",
    "t700",
    "t850",
    "t925",
    "t1000",
    "r50",
    "r100",
    "r150",
    "r200",
    "r250",
    "r300",
    "r400",
    "r500",
    "r600",
    "r700",
    "r850",
    "r925",
    "r1000",
]


class FCNv2(torch.nn.Module, AutoModel, PrognosticMixin):
    """FourCastNet v2 global prognostic model. Consists of a single model with a
    time-step size of 6 hours. FourCastNet v2 operates on a [720 x 1440] equirectangular
    grid with 73 variables.

    Note
    ----
    For additional information see the following resources:

    - https://arxiv.org/abs/2306.03838
    - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/modulus_fcnv2_sm

    Parameters
    ----------
    core_model : torch.nn.Module
        Core PyTorch model with loaded weights
    center : torch.Tensor
        Model center normalization tensor of size [73]
    scale : torch.Tensor
        Model scale normalization tensor of size [73]
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
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(90, -90, 1440, endpoint=False),
        }
    )

    output_coords = OrderedDict(
        {
            "lead_time": np.array([datetime.timedelta(hours=6)]),
            "variable": np.array(VARIABLES),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(90, -90, 1440, endpoint=False),
        }
    )

    @staticmethod
    def _fix_state_dict_keys(
        state_dict: dict[str, torch.Tensor], add_module: bool = False
    ) -> dict[str, torch.Tensor]:
        """Add or remove 'module.' from state_dict keys

        Parameters
        ----------
        state_dict : Dict
            Model state_dict
        add_module : bool, optional
            If True, will add 'module.' to keys, by default False

        Returns
        -------
        Dict
            Model state_dict with fixed keys
        """
        fixed_state_dict = {}
        for key, value in state_dict.items():
            if add_module:
                new_key = "module." + key
            else:
                new_key = key.replace("module.", "")
            fixed_state_dict[new_key] = value
        return fixed_state_dict

    @classmethod
    def load_default_package(cls) -> Package:
        return Package("ngc://model/nvidia/modulus/modulus_fcnv2_sm@v0.2")

    @classmethod
    def load_model(
        cls,
        package: Package,
    ) -> PrognosticModel:
        # TODO: Remmove Hack with Yparams
        config_path = pathlib.Path(__file__).parent / "nn" / "fcnv2" / "sfnonet.yaml"
        params = fcnv2.YParams(config_path.as_posix(), "sfno_73ch")

        core_model = fcnv2.FourierNeuralOperatorNet(params)

        local_center = torch.Tensor(np.load(package.get("global_means.npy")))
        local_std = torch.Tensor(np.load(package.get("global_stds.npy")))

        weights_path = package.get("weights.tar")
        weights = torch.load(weights_path, map_location="cpu")
        fixed_weights = cls._fix_state_dict_keys(
            weights["model_state"], add_module=False
        )
        core_model.load_state_dict(fixed_weights)

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
