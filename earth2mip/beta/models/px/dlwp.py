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

import datetime
from collections import OrderedDict
from typing import Any, Iterator

import modulus
import numpy as np
import torch
import xarray
from modulus.utils.zenith_angle import cos_zenith_angle

from earth2mip.beta.models.auto import AutoModel, Package
from earth2mip.beta.models.px.base import PrognosticModel
from earth2mip.beta.models.px.utils import PrognosticMixin
from earth2mip.beta.utils import handshake_coord, handshake_dim, handshake_size

VARIABLES = ["t850", "z1000", "z700", "z500", "z300", "tcwv", "t2m"]


class DLWP(torch.nn.Module, AutoModel, PrognosticMixin):
    def __init__(
        self,
        core_model: torch.nn.Module,
        landsea_mask: torch.Tensor,
        topographic_height: torch.Tensor,
        latgrid: torch.Tensor,
        longrid: torch.Tensor,
        ll_to_cs_mapfile_path: str,
        cs_to_ll_mapfile_path: str,
        center: torch.Tensor,
        scale: torch.Tensor,
    ):
        super().__init__()
        self.model = core_model
        self.ll_to_cs_mapfile_path = ll_to_cs_mapfile_path
        self.cs_to_ll_mapfile_path = cs_to_ll_mapfile_path
        self.register_buffer("landsea_mask", landsea_mask)
        self.register_buffer("topographic_height", topographic_height)
        self.register_buffer("latgrid", latgrid)
        self.register_buffer("longrid", longrid)
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)

        # Load map weights
        input_map_wts = xarray.open_dataset(ll_to_cs_mapfile_path)
        output_map_wts = xarray.open_dataset(cs_to_ll_mapfile_path)

        i = input_map_wts.row.values - 1
        j = input_map_wts.col.values - 1
        data = input_map_wts.S.values
        self.register_buffer("M_in", torch.sparse_coo_tensor(np.array((i, j)), data))

        i = output_map_wts.row.values - 1
        j = output_map_wts.col.values - 1
        data = output_map_wts.S.values
        self.register_buffer("M_out", torch.sparse_coo_tensor(np.array((i, j)), data))

    input_coords = OrderedDict(
        {
            "lead_time": np.array(
                [datetime.timedelta(hours=-6), datetime.timedelta(0)]
            ),
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

    @classmethod
    def load_default_package(cls) -> Package:
        return Package("ngc://model/nvidia/modulus/modulus_dlwp@v0.2")

    @classmethod
    def load_model(
        cls,
        package: Package,
    ) -> PrognosticModel:
        # load static datasets
        lsm = xarray.open_dataset(package.get("land_sea_mask_rs_cs.nc"))["lsm"].values
        topographic_height = xarray.open_dataset(package.get("geopotential_rs_cs.nc"))[
            "z"
        ].values
        latlon_grids = xarray.open_dataset(package.get("latlon_grid_field_rs_cs.nc"))
        latgrid = latlon_grids["latgrid"].values
        longrid = latlon_grids["longrid"].values
        # load maps
        ll_to_cs_mapfile_path = package.get("map_LL721x1440_CS64.nc")
        cs_to_ll_mapfile_path = package.get("map_CS64_LL721x1440.nc")

        core_model = modulus.Module.from_checkpoint(package.get("dlwp.mdlus"))

        center = torch.Tensor(np.load(package.get("global_means.npy")))
        scale = torch.Tensor(np.load(package.get("global_stds.npy")))

        return cls(
            core_model,
            landsea_mask=torch.Tensor(lsm),
            topographic_height=torch.Tensor(topographic_height),
            latgrid=torch.Tensor(latgrid),
            longrid=torch.Tensor(longrid),
            ll_to_cs_mapfile_path=ll_to_cs_mapfile_path,
            cs_to_ll_mapfile_path=cs_to_ll_mapfile_path,
            center=center,
            scale=scale,
        )

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
            Coordinate system, should have dimensions [time, lead_time, variable, lat, lon]

        Yields
        ------
        Iterator[Tuple[torch.Tensor, OrderedDict[str, np.ndarray], Any]]
            Time iterator
        """
        handshake_dim(coords, "lon", 4)
        handshake_dim(coords, "lat", 3)
        handshake_dim(coords, "variable", 2)
        handshake_dim(coords, "lead_time", 1)
        handshake_dim(coords, "time", 0)
        handshake_coord(coords, self.input_coords, "lon")
        handshake_coord(coords, self.input_coords, "lat")
        handshake_coord(coords, self.input_coords, "variable")
        handshake_size(coords, "lead_time", 2)

        yield from self._dlwp_iterator(x, coords)

    def _prepare_input(
        self, input: torch.Tensor, coords: OrderedDict[str, np.ndarray]
    ) -> torch.Tensor:
        device = input.device
        dtype = input.dtype

        bs, t, chan = input.shape[0], input.shape[1], input.shape[2]
        input = input.reshape(bs * t * chan, -1) @ self.M_in.T
        input = input.reshape(bs, t, chan, 6, 64, 64)
        input_list = list(torch.split(input, 1, dim=1))
        input_list = [tensor.squeeze(1) for tensor in input_list]
        repeat_vals = (input.shape[0], -1, -1, -1, -1)  # repeat along batch dimension
        for i in range(len(input_list)):
            tisr = np.maximum(
                cos_zenith_angle(
                    coords["time"][i]
                    - datetime.timedelta(hours=6 * (t - 1))
                    + datetime.timedelta(hours=6 * i),
                    self.longrid,
                    self.latgrid,
                ),
                0,
            ) - (
                1 / np.pi
            )  # subtract mean value
            tisr = (
                torch.tensor(tisr, dtype=dtype)
                .to(device)
                .unsqueeze(dim=0)
                .unsqueeze(dim=0)
            )  # add channel and batch size dimension
            tisr = tisr.expand(*repeat_vals)  # TODO - find better way to batch TISR
            input_list[i] = torch.cat(
                (input_list[i], tisr), dim=1
            )  # concat along channel dim

        input_model = torch.cat(
            input_list, dim=1
        )  # concat the time dimension into channels

        lsm_tensor = torch.tensor(self.lsm, dtype=dtype).to(device).unsqueeze(dim=0)
        lsm_tensor = lsm_tensor.expand(*repeat_vals)
        topographic_height_tensor = (
            torch.tensor((self.topographic_height - 3.724e03) / 8.349e03, dtype=dtype)
            .to(device)
            .unsqueeze(dim=0)
        )
        topographic_height_tensor = topographic_height_tensor.expand(*repeat_vals)

        input = torch.cat((input_model, lsm_tensor, topographic_height_tensor), dim=1)
        return input

    def _prepare_output(self, output: torch.Tensor) -> torch.Tensor:
        output = torch.split(output, output.shape[1] // 2, dim=1)
        output = torch.stack(output, dim=1)  # add time dimension back in

        output = output.reshape(output.shape[0], 2, output.shape[2], -1) @ self.M_out.T
        output = output.reshape(output.shape[0], 2, output.shape[2], 721, 1440)

        return output

    def _dlwp_iterator(
        self, x: torch.Tensor, coords: OrderedDict[str, np.ndarray]
    ) -> Iterator[tuple[torch.Tensor, OrderedDict[str, np.ndarray], Any]]:
        """Yield (time, unnormalized data, restart) tuples"""
        coords = coords.copy()
        out_coord = coords.copy()
        del out_coord["lead_time"]
        x = (x - self.center) / self.scale

        while True:
            # Front hook
            x, coords = self.front_hook(x, coords)

            # Slightly inefficient transforming every prediction, but allows for normal
            # hooks, can likley be optimized
            x = self._prepare_input(x, coords)
            y = self.model(x)
            x = self._prepare_output(y)

            coords["time"] += datetime.timedelta(hours=12)

            # Rear hook
            x, coords = self.rear_hook(x, coords)

            out = self.scale * x + self.center
            out_coord["time"] += datetime.timedelta(hours=6)
            yield out[:, 0], out_coord, None

            out_coord["time"] += datetime.timedelta(hours=6)
            yield out[:, 1], out_coord, None
