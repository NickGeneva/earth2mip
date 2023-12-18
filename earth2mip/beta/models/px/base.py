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

from collections import OrderedDict
from typing import Any, Iterator, Protocol, Tuple, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class PrognosticModel(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        coords: OrderedDict[str, np.ndarray],
    ) -> Iterator[Tuple[torch.Tensor, OrderedDict[str, np.ndarray], Any]]:
        """Creates a iterator that performs time-integration with the prognostic model

            Parameters
        ----------
        x : torch.Tensor
            Input tensor intended to apply perturbation on
        coords : OrderedDict[str, np.ndarray]
            Ordered dict representing coordinate system that discribes the tensor

        Returns
        -------
        Iterator[Tuple[torch.Tensor, OrderedDict[str, np.ndarray], Any]]:
            Iterator object (yeild some method), that generates timesteps with
            coordinate system.
        """
        pass

    @property
    def input_coords(self) -> OrderedDict[str, np.ndarray]:
        """Input coordinate system of prognostic model, time dimension should contain
        time-delta objects

        Returns
        -------
        OrderedDict[str, np.ndarray]
            Coordinate system dictionary
        """
        pass

    @property
    def output_coords(self) -> OrderedDict[str, np.ndarray]:
        """Ouput coordinate system of prognostic model, time dimension should contain
        time-delta objects

        Returns
        -------
        OrderedDict[str, np.ndarray]
            Coordinate system dictionary
        """
        pass
