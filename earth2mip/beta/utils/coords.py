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
from typing import Optional

import numpy as np


def handshake_dim(
    input_coords: OrderedDict[str, np.ndarray],
    required_dim: str,
    required_index: Optional[int] = None,
) -> None:
    """Simple check to see if coordinate system has a dimension in a particular index

    Parameters
    ----------
    input_coords : OrderedDict[str, np.ndarray]
        Input coordinate system to validate
    required_dim : str
        Required dimension (name of coordinate)
    required_index : optional[int], optional
        Required index of dimension if needed, by default None

    Raises
    ------
    KeyError
        If required dimension is not found in the input coordinate system
    ValueError
        If the required index is outside the dimensionality of the input coordinate system
    ValueError
        If dimension is not in the required index
    """

    if required_dim not in input_coords:
        raise KeyError(
            f"Required dimension {required_dim} not found in input coordinates"
        )

    input_dims = list(input_coords.keys())

    if required_index is None:
        return

    try:
        input_dims[required_index]
    except IndexError:
        raise ValueError(
            f"Required index {required_index} outside dimensionality of input coordinate system of {len(input_dims)}"
        )

    if input_dims[required_index] != required_dim:
        raise ValueError(
            f"Required dimension {required_dim} not found in the required index {required_index} in dim list {input_dims}"
        )


def handshake_coord(
    input_coords: OrderedDict[str, np.ndarray],
    target_coords: OrderedDict[str, np.ndarray],
    required_dim: str,
) -> None:
    """Simple check to see if the required dimensions have the same coordiante system

    Parameters
    ----------
    input_coords : OrderedDict[str, np.ndarray]
        Input coordinate system to validate
    target_coords : OrderedDict[str, np.ndarray]
        Target coordinate system
    required_dim : str
        Required dimension (name of coordinate)

    Raises
    ------
    KeyError
        If required dim is not present in coordinate systems
    ValueError
        If coordinates of required dimensions don't match
    """
    if required_dim not in input_coords:
        raise KeyError(
            f"Required dimension {required_dim} not found in input coordinates"
        )

    if required_dim not in target_coords:
        raise KeyError(
            f"Required dimension {required_dim} not found in target coordinates"
        )

    if not np.all(input_coords[required_dim] == target_coords[required_dim]):
        raise ValueError(
            f"Coordinate systems for required dim {required_dim} are not the same"
        )


def handshake_size(
    input_coords: OrderedDict[str, np.ndarray],
    required_dim: str,
    required_size: int,
) -> None:
    """Simple check to see if a coordinate system of a given dimension is a required
    size

    Parameters
    ----------
    input_coords : OrderedDict[str, np.ndarray]
        Input coordinate system to validate
    required_dim : str
        Required dimension (name of coordinate)
    required_size : int
        Required coordinate system size

    Note
    ----
    Presently assumes coordinate system of given dimension is 1D

    Raises
    ------
    KeyError
        If required dim is not present in input coordinate system
    ValueError
        If required dimension is not of required size
    """

    if required_dim not in input_coords:
        raise KeyError(
            f"Required dimension {required_dim} not found in input coordinates"
        )

    if input_coords[required_dim].shape[0] != required_size:
        raise ValueError(
            f"Coordinate size for required dim {required_dim} is not of size {required_size}"
        )
