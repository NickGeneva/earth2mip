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
from typing import List, Protocol, Union, runtime_checkable

import xarray as xr


@runtime_checkable
class DataSource(Protocol):
    """Data source interface."""

    def __call__(
        self,
        t: Union[datetime.datetime, List[datetime.datetime]],
        channel: Union[str, List[str]],
    ) -> xr.DataArray:
        """Function to get data.

        Parameters
        ----------
        t : datetime.datetime or List[datetime.datetime]
            Timestamps to return data for.
        channel : str or List[str]
            Strings or list of strings that refer to the
            channel/variables to return.

        Returns
        -------
        xr.DataArray
            An xarray data-array with the dimensions [time, channel, ....]. The coords
            should be provided. Time coordinate should be a datetime array and the
            channel coordinate should be array of strings with E2-MIP channel ids.
        """
        pass
