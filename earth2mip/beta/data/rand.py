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
from typing import List, Union

import numpy as np
import xarray as xr


class Random:
    """A randomly generated normaally distributed data. Primarily useful for testing.

    Parameters
    ----------
    lat : Union[List[float], np.array]
        1D list or numpy array of latitude coordinates
    lon : Union[List[float], np.array]
        1D list or numpy array of longitude coordinates
    """

    def __init__(
        self,
        lat: Union[List[float], np.array],
        lon: Union[List[float], np.array],
    ):
        if isinstance(lat, np.ndarray) and not lat.ndim == 1:
            raise ValueError("Latitude array must be 1D")
        if isinstance(lon, np.ndarray) and not lon.ndim == 1:
            raise ValueError("Longitude array must be 1D")

        self.lat = lat
        self.lon = lon

    def __call__(
        self,
        time: Union[datetime.datetime, List[datetime.datetime]],
        variable: Union[str, List[str]],
    ) -> xr.DataArray:
        """Retrieve random gaussian data.

        Parameters
        ----------
        time : Union[datetime.datetime, List[datetime.datetime]]
            Timestamps to return data for.
        variable : Union[str, List[str]]
            Strings or list of strings that refer to variables to return.

        Returns
        -------
        xr.DataArray
            Random data array
        """

        if isinstance(variable, str):
            variable = [variable]

        if isinstance(time, datetime.datetime):
            time = [time]

        da = xr.DataArray(
            data=np.random.randn(
                len(time), len(variable), len(self.lat), len(self.lon)
            ),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": time,
                "variable": variable,
                "lat": self.lat,
                "lon": self.lon,
            },
        )

        return da
