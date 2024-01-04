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
Pangu Weather adapter

adapted from https://raw.githubusercontent.com/ecmwf-lab/ai-models-panguweather/main/ai_models_panguweather/model.py

# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
"""
# %%
import datetime
import os
from collections import OrderedDict
from typing import Any, Iterator

import numpy as np
import onnxruntime as ort
import torch

from earth2mip.beta.models.auto import AutoModel, Package
from earth2mip.beta.models.px.base import PrognosticModel
from earth2mip.beta.models.px.utils import PrognosticMixin
from earth2mip.beta.utils import handshake_coord, handshake_dim

VARIABLES = [
    "z1000",
    "z925",
    "z850",
    "z700",
    "z600",
    "z500",
    "z400",
    "z300",
    "z250",
    "z200",
    "z150",
    "z100",
    "z50",
    "q1000",
    "q925",
    "q850",
    "q700",
    "q600",
    "q500",
    "q400",
    "q300",
    "q250",
    "q200",
    "q150",
    "q100",
    "q50",
    "t1000",
    "t925",
    "t850",
    "t700",
    "t600",
    "t500",
    "t400",
    "t300",
    "t250",
    "t200",
    "t150",
    "t100",
    "t50",
    "u1000",
    "u925",
    "u850",
    "u700",
    "u600",
    "u500",
    "u400",
    "u300",
    "u250",
    "u200",
    "u150",
    "u100",
    "u50",
    "v1000",
    "v925",
    "v850",
    "v700",
    "v600",
    "v500",
    "v400",
    "v300",
    "v250",
    "v200",
    "v150",
    "v100",
    "v50",
    "msl",
    "u10m",
    "v10m",
    "t2m",
]


class PanguBase(torch.nn.module, PrognosticMixin):
    """Pangu weather base class. Contains common methods"""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("center", torch.zeros(len(VARIABLES)))
        self.register_buffer("scale", torch.ones(len(VARIABLES)))

    input_coords = OrderedDict(
        {
            "lead_time": np.array([datetime.timedelta(0)]),
            "variable": np.array(VARIABLES),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(90, -90, 1440, endpoint=False),
        }
    )

    def _forward_step(
        self,
        x: torch.Tensor,
        coords: OrderedDict[str, np.ndarray],
        ort: ort.InferenceSession,
    ) -> tuple[torch.Tensor, OrderedDict[str, np.ndarray]]:

        # Front hook
        x, coords = self.front_hook(x, coords)
        # Forward pass
        x = self._onnx_forward(ort, x)
        coords["time"] = coords["time"] + self.output_coords["lead_time"]
        # Rear hook
        x, coords = self.rear_hook(x, coords)
        return x, coords

    @staticmethod
    def _onnx_forward(
        ort_session: ort.InferenceSession, x: torch.Tensor
    ) -> torch.Tensor:

        # from https://onnxruntime.ai/docs/api/python/api_summary.html
        binding = ort_session.io_binding()

        def bind_input(name: str, x: torch.Tensor) -> None:
            x = x.contiguous()
            binding.bind_input(
                name=name,
                device_type="cuda",
                # device_id=self.device_index,
                element_type=np.float32,
                shape=tuple(x.shape),
                buffer_ptr=x.data_ptr(),
            )

        def bind_output(name: str, like: torch.Tensor) -> torch.Tensor:
            x = torch.empty_like(like).contiguous()
            binding.bind_output(
                name=name,
                device_type="cuda",
                # device_id=self.device_index,
                element_type=np.float32,
                shape=tuple(x.shape),
                buffer_ptr=x.data_ptr(),
            )
            return x

        pl_shape = (5, 13, 721, 1440)
        nchan = pl_shape[0] * pl_shape[1]
        pl = x[:, :nchan]
        surface = x[:, nchan:]
        fields_pl = pl.resize(*pl_shape)
        fields_sfc = surface[0]

        bind_input("input", fields_pl)
        bind_input("input_surface", fields_sfc)
        output = bind_output("output", like=fields_pl)
        output_sfc = bind_output("output_surface", like=fields_sfc)
        ort_session.run_with_iobinding(binding)

        return torch.cat(
            [
                output.resize(1, nchan, 721, 1440),
                output_sfc.resize(1, x.size(1) - nchan, 721, 1440),
            ],
            dim=1,
        )


class Pangu_Single(PanguBase):
    """Pangu weather single model. For use with individual models

    Parameters
    ----------
    ort : ort.InferenceSession
        Pangu ONNX runtime session
    time_delta : datetime.timedelta, optional
        Time step size of provided model, by default datetime.timedelta(hours=24)
    """

    def __init__(
        self,
        ort: ort.InferenceSession,
        time_delta: datetime.timedelta = datetime.timedelta(hours=24),
    ):
        super().__init__()
        self.ort = ort.InferenceSession

        self.output_coords = OrderedDict(
            {
                "lead_time": np.array([time_delta]),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(90, -90, 1440, endpoint=False),
            }
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

        yield from self._pangu_single_iterator(x, coords)

    def _pangu_single_iterator(
        self, x: torch.Tensor, coords: OrderedDict[str, np.ndarray]
    ) -> Iterator[tuple[torch.Tensor, OrderedDict[str, np.ndarray], Any]]:

        coords = coords.copy()
        x = (x - self.center) / self.scale
        while True:
            # Forward step
            x, coords = self._forward_step(x, coords, self.ort)
            # Denormalize
            out = self.scale * x + self.center
            yield out, coords, None


class Pangu_Double(PanguBase):
    def __init__(self, ort_6hr: ort.InferenceSession, ort_24hr: ort.InferenceSession):
        """Dual model Pangu weather

        Parameters
        ----------
        ort_6hr : ort.InferenceSession
            6 hours Pangu ONNX runtime session
        ort_24hr : ort.InferenceSession
            24 hour Pangu ONNX runtime session
        """
        super().__init__()
        self.ort_6hr = ort_6hr
        self.ort_24hr = ort_24hr

    output_coords = OrderedDict(
        {
            "lead_time": np.array([datetime.timedelta(hours=6)]),
            "variable": np.array(VARIABLES),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(90, -90, 1440, endpoint=False),
        }
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

        yield from self._pangu_iterator(x, coords)

    def _pangu_iterator(
        self, x: torch.Tensor, coords: OrderedDict[str, np.ndarray]
    ) -> Iterator[tuple[torch.Tensor, OrderedDict[str, np.ndarray], Any]]:

        coords = coords.copy()
        x = (x - self.center) / self.scale
        while True:
            x0 = x.copy()
            coords0 = coords.copy()
            # Perform 6 hour steps
            for i in range(3):
                # Forward step
                x, coords = self._forward_step(x, coords, self.ort_6hr)
                # Denormalize
                out = self.scale * x + self.center
                yield out, coords, None

            # 24hr Forward step
            x, _ = self._forward_step(x0, coords0, self.ort_24hr)
            coords["time"] = coords["time"] + self.output_coords["lead_time"]
            # Denormalize
            out = self.scale * x + self.center
            yield out, coords, None


class Pangu(AutoModel):
    """Pangu weather loader class. Contains methods for loading and initializing
    different Pangu weather models
    """

    @classmethod
    def load_default_package(cls) -> Package:
        return Package("https://get.ecmwf.int/repository/test-data/ai-models")

    @classmethod
    def load_model(
        cls,
        package: Package,
    ) -> PrognosticModel:

        ort_file = package.get("pangu_weather_6.onnx")
        ort_session_6 = cls._create_ort_session(ort_file)
        ort_file = package.get("pangu_weather_24.onnx")
        ort_session_24 = cls._create_ort_session(ort_file)

        return Pangu_Double(ort_session_6, ort_session_24)

    @classmethod
    def load_model_6hr(
        cls,
        package: Package,
    ) -> PrognosticModel:

        ort_file = package.get("pangu_weather_6.onnx")
        ort_session = cls._create_ort_session(ort_file)
        dt = datetime.timedelta(hours=6)

        return Pangu_Single(ort_session, time_delta=dt)

    @classmethod
    def load_model_24hr(
        cls,
        package: Package,
    ) -> PrognosticModel:

        ort_file = package.get("pangu_weather_24.onnx")
        ort_session = cls._create_ort_session(ort_file)
        dt = datetime.timedelta(hours=24)

        return Pangu_Single(ort_session, time_delta=dt)

    @staticmethod
    def _create_ort_session(ort_file: str) -> ort.InferenceSession:
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False
        options.intra_op_num_threads = 1

        # That will trigger a FileNotFoundError
        device_index = torch.cuda.current_device()
        os.stat(ort_file)
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": device_index,
                },
            ),
            "CPUExecutionProvider",
        ]

        ort_session = ort.InferenceSession(
            ort_file,
            sess_options=options,
            providers=providers,
        )

        return ort_session
