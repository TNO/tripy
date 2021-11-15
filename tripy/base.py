"""
 This module contains utility functions to construct the likelihood function for the
 parameter_estimation module.
 It is a non-essential but a convenience module.

 It supports the construction of a selected set of likelihood types we often
 encounter/use.

 Current limitations (not by design):
    - only within group correlation is handled
    - correlation over time has no relation to correlation over space (although they
        are combined at the end)

 TODO:
  * decorator type error/warning if there is no measurement points yet available
  * additional visualization of the input:
    - correlations
    - standard deviations
  * document functions
  * extend: add the possibility to the time dimension to have different correlation
    per space group!
  * extend: add option for between group correlation
  * generalize the code a bit: the same code should work for measurement uncertainty
    and model uncertainty -> can contain correlation
  * maybe try numba for the likelihood eval?

"""

import logging
from typing import Callable, Iterable, Type, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.recfunctions import stack_arrays
from tabulate import tabulate

from tripy.kernels import kernel
from tripy.utils import correlation_matrix, grow_mx


class MeasurementSpaceTimePoints:
    """
    A convenience class for defining **one** source/type of uncertainty and building
    a covariance matrix and correlation matrix.

    **One** source/type means that one object of this class is to be used for a
    single uncertainty source, e.g. measurement uncertainty and another one for model
    uncertainty. Since these two uncertainty sources belong to the same
    points/sensors on the structure you should reuse one object when creating the other,
    e.g. the sensor naming and location.

    There are two axes:
        Measurement space: sensors at different physical location or measuring
            different physical quantities (combined space and measured quantity space).
        Measurement time: sensor measurements at different times, can mean load
            cases, construction stages, etc.

    There is a third important characteristics: measured physical quantity, such as
    strain or translation. Though that does not qualify to have its own axis.
    """

    def __init__(self):
        self.measurement_space_points = None
        self.measurement_time_points = None
        self.measurement_space_group_correlations = None
        self.measurement_time_group_correlations = None
        self.cov_mx = None
        self.corr_mx = None

    def __str__(self):
        # ..............................................................................
        # Space points
        # ..............................................................................
        measurement_space_points = self.measurement_space_points
        if measurement_space_points is not None:
            repr_space = tabulate(
                measurement_space_points,
                headers=measurement_space_points.dtype.names,
                showindex=True,
                floatfmt=".2f",
            )
        else:
            repr_space = (
                "empty (You can add measurement space points using the "
                "`add_measurement_space_points` method.)"
            )

        # ..............................................................................
        # Time points
        # ..............................................................................
        measurement_time_points = self.measurement_time_points
        if measurement_time_points is not None:
            repr_time = tabulate(
                measurement_time_points,
                headers=measurement_time_points.dtype.names,
                showindex=True,
                floatfmt=".2f",
            )
        else:
            repr_time = (
                "empty (You can add measurement time points using the "
                "`add_measurement_time_points` method.)"
            )

        # ..............................................................................
        # Combine
        # ..............................................................................
        repr_comb = (
            "Measurement space points \n" + "=" * 30 + f"\n{repr_space}\n\n"
            "Measurement time points \n" + "=" * 30 + f"\n{repr_time}"
        )
        return repr_comb

    def add_measurement_space_points(
        self,
        coord_mx: Union[np.ndarray, Iterable, float, int],
        standard_deviation: Union[np.ndarray, Iterable, float, int] = 0,
        group: str = "ungrouped",
        sensor_name: str = "sensor",
        sensor_number: Union[np.ndarray, Iterable, float, int] = None,
    ) -> None:
        """
        Add measurement space points.

        Args:
            coord_mx: matrix with coordinates of the measurement point(s).
            standard_deviation: standard deviation of the model uncertainty.
            group: measurement space point group. elastic_mod.g. "strain" or
                "translation". The default group is "ungrouped".
            sensor_name: sensor name. The default name is "sensor".
            sensor_number: sensor number. If none is provided a progressive
                numeration is assigned by default.

        """
        # ..............................................................................
        # Check input
        # ..............................................................................
        self._check_space_point_input_consistency(
            coord_mx=coord_mx,
            standard_deviation=standard_deviation,
            group=group,
            sensor_name=sensor_name,
            sensor_number=sensor_number,
        )

        # ..............................................................................
        # Pre-process input -> bring into a fixed format (input is flexible)
        # ..............................................................................
        new_measurement_space_points = self._pre_process_space_point_input(
            coord_mx=coord_mx,
            standard_deviation=standard_deviation,
            group=group,
            sensor_name=sensor_name,
            sensor_number=sensor_number,
        )

        # ..............................................................................
        # Append sensors to existing ones
        # ..............................................................................
        if self.measurement_space_points is None:
            self.measurement_space_points = new_measurement_space_points
        else:
            self.measurement_space_points = stack_arrays(
                (self.measurement_space_points, new_measurement_space_points),
                asrecarray=True,
                usemask=False,
                autoconvert=True,
            )

    def add_measurement_time_points(
        self,
        coord_vec: Union[np.ndarray, Iterable, float, int],
        group: str = "ungrouped",
    ) -> None:
        """
        Add measurement time point(s).

        Args:
            coord_vec: time coordinate for each measurement time point.
            group: measurement time point group. elastic_mod.g. "strain" or
                "translation".

        """

        # ..............................................................................
        # Check input
        # ..............................................................................
        self._check_time_point_input_consistency(
            coord_vec=coord_vec, group=group,
        )

        # ..............................................................................
        # Pre-process input -> bring into a fixed format (input is flexible)
        # ..............................................................................
        new_measurement_time_points = self._pre_process_time_point_input(
            coord_vec=coord_vec, group=group,
        )

        # ..............................................................................
        # Append sensors to existing ones
        # ..............................................................................
        if self.measurement_time_points is None:
            self.measurement_time_points = new_measurement_time_points
        else:
            self.measurement_time_points = stack_arrays(
                (self.measurement_time_points, new_measurement_time_points),
                asrecarray=True,
                usemask=False,
                autoconvert=True,
            )

    def add_measurement_space_within_group_correlation(
        self,
        group: str,
        correlation_func: Union[Type[kernel], Callable[[np.ndarray], np.ndarray]],
        **kwargs,
    ):
        """
        Assign a correlation function to the specified measurement space point group.

        Args:
            group: measurement space point group. elastic_mod.g. "strain" or
                "translation".
            correlation_func: correlation function to be assigned.
            kwargs: Optional parameters to be passed to correlation_func

        """

        measurement_space_points = self.measurement_space_points

        if measurement_space_points is None:
            raise RuntimeError(
                "Before adding correlation you first need to add points to correlate "
                "(`add_measurement_space_points`)!"
            )

        groups = np.unique(measurement_space_points.group)

        if not isinstance(group, (str, type(None))):
            raise TypeError(
                "`group` should be a (single) string. You need to define correlations "
                "one at a time."
            )
        if group not in groups:
            raise ValueError(
                f"The provided `group` value ({group}) is not amongst the existing "
                f"groups in measurement points."
            )

        # If the supplied object has the required methods, assume it is a kernel.
        # Otherwise assume it is a callable.
        obj_dir = set(dir(correlation_func))
        required_methods = {"corr", "eval"}

        if required_methods <= obj_dir:
            ms_pts = self.measurement_space_points[
                self.measurement_space_points.group == group
            ]
            coord_x1 = ms_pts.coord_x1
            coord_x2 = ms_pts.coord_x2
            coord_x3 = ms_pts.coord_x3
            coords = np.vstack((coord_x1, coord_x2, coord_x3)).transpose()
            std_dev = ms_pts.standard_deviation
            kernel_obj = correlation_func(coords, std_dev, **kwargs)
        else:
            print(
                f"Object {correlation_func.__name__} does not have methods"
                f" {required_methods}. Assuming it is a callable."
            )
            kernel_obj = correlation_func

        if self.measurement_space_group_correlations is None:
            self.measurement_space_group_correlations = {group: kernel_obj}
        else:
            self.measurement_space_group_correlations.update({group: kernel_obj})

    def add_measurement_time_within_group_correlation(
        self,
        group: str,
        correlation_func: Union[Type[kernel], Callable[[np.ndarray], np.ndarray]],
        **kwargs,
    ):
        """
        Assign a correlation function to the specified measurement time points group.

        Args:
            group: measurement time point group. elastic_mod.g. "strain" or
                "translation".
            correlation_func: correlation function to be assigned.
            kwargs: Optional parameters to be passed to correlation_func

        """

        measurement_time_points = self.measurement_time_points

        groups = np.unique(measurement_time_points.group)

        if measurement_time_points is None:
            raise RuntimeError(
                "Before adding correlation you first need to add points to correlate "
                "(`add_measurement_time_points`)!"
            )
        if not isinstance(group, (str, type(None))):
            raise TypeError(
                "`group` should be a (single) string. You need to define correlations "
                "one at a time."
            )
        if group not in groups:
            raise ValueError(
                f"The provided `group` value ({group}) is not amongst the existing "
                f"groups in measurement points."
            )

        # If the supplied object has the required methods, assume it is a kernel.
        # Otherwise assume it is a callable.
        obj_dir = set(dir(correlation_func))
        required_methods = {"corr", "eval"}

        if required_methods <= obj_dir:
            ms_pts = self.measurement_time_points[
                self.measurement_time_points.group == group
            ]
            coord_t = ms_pts.coord_t
            std_dev = np.ones(len(coord_t))
            kernel_obj = correlation_func(coord_t.reshape(-1, 1), std_dev, **kwargs)
        else:
            print(
                f"Object {correlation_func.__name__} does not have methods"
                f"{required_methods}. Assuming it is a callable."
            )
            kernel_obj = correlation_func

        if self.measurement_time_group_correlations is None:
            self.measurement_time_group_correlations = {group: kernel_obj}
        else:
            self.measurement_time_group_correlations.update({group: kernel_obj})

    def compile_covariance_matrix(self) -> np.ndarray:
        """
        Compile the covariance matrix (`k_cov_mx`) considering the space
        and time points previously added and the assigned correlation functions.
        After calling this method also the correlation matrix is available in the
        `corr_mx` attribute.

        Returns:
            k_cov_mx: covariance matrix, shape [T*S, T*S].

        """
        # [Jan] TODO: Time this function.
        # [Jan] NOTES: It seems that this function is not really designed for performing
        # parameter estimation including the uncertainty parameters (i.e. optimized for
        # speed). Would the sensor/measurement positions change throughout the parameter
        # estimation? If not then the pre-processing could be performed only once after
        # formulating the problem.

        # ..............................................................................
        # Pre-processing
        # ..............................................................................
        space_groups_all = self.measurement_space_points.group
        space_standard_deviation = self.measurement_space_points.standard_deviation
        space_coord_mx = np.vstack(
            (
                self.measurement_space_points.coord_x1,
                self.measurement_space_points.coord_x2,
                self.measurement_space_points.coord_x3,
            )
        ).T
        space_group_correlations = self.measurement_space_group_correlations
        n_space_point = space_coord_mx.shape[0]

        if self.measurement_time_points is None:
            self.add_measurement_time_points(coord_vec=0)

        time_groups_all = self.measurement_time_points.group
        time_coord_vec = self.measurement_time_points.coord_t
        time_group_correlations = self.measurement_time_group_correlations
        n_time_point = time_coord_vec.shape[0]

        space_corr_mx_no_time = np.diag(np.ones(n_space_point))
        time_corr_mx_no_space = np.diag(np.ones(n_time_point))

        # ------------------------------------------------------------------------------
        # Compilation
        # ------------------------------------------------------------------------------
        # ..............................................................................
        # Space correlation
        # ..............................................................................
        for space_group in np.unique(space_groups_all):
            if space_group_correlations is not None:
                if space_group in space_group_correlations:
                    idx_space_group = np.argwhere(
                        space_groups_all == space_group
                    ).squeeze()
                    space_coord_mx_group = space_coord_mx[idx_space_group]
                    correlation_func = space_group_correlations[space_group]

                    if callable(correlation_func):
                        space_corr_mx_group = correlation_matrix(
                            space_coord_mx_group, correlation_func=correlation_func
                        )
                    else:
                        space_corr_mx_group = correlation_func.corr()

                    idx_mesh = np.array(np.meshgrid(idx_space_group, idx_space_group))
                    idx_comb = idx_mesh.T.reshape(-1, 2)
                    space_corr_mx_no_time[
                        idx_comb[:, 0], idx_comb[:, 1]
                    ] = space_corr_mx_group.ravel()

        # the full space corr mx
        space_corr_mx = np.tile(space_corr_mx_no_time, (n_time_point, n_time_point))

        # standard deviation
        standard_deviation = np.tile(space_standard_deviation, n_time_point)
        standard_deviation_diag_mx = np.diag(standard_deviation)

        # ..............................................................................
        # Time correlation
        # ..............................................................................
        for time_group in np.unique(time_groups_all):
            if time_group_correlations is not None:
                if time_group in time_group_correlations:
                    idx_time_group = np.argwhere(time_groups_all == time_group)
                    time_coord_vec_group = time_coord_vec[idx_time_group]
                    correlation_func = time_group_correlations[time_group]

                    if callable(correlation_func):
                        time_corr_mx_group = correlation_matrix(
                            time_coord_vec_group, correlation_func=correlation_func
                        )
                    else:
                        time_corr_mx_group = correlation_func.corr()

                    idx_mesh = np.array(np.meshgrid(idx_time_group, idx_time_group))
                    idx_comb = idx_mesh.T.reshape(-1, 2)
                    time_corr_mx_no_space[
                        idx_comb[:, 0], idx_comb[:, 1]
                    ] = time_corr_mx_group.ravel()

        # [Jan] Could this part be replaced by a np.kron(time_corr_mx, space_corr_mx)?

        # the full time corr mx
        time_corr_mx = grow_mx(
            seed_mx=time_corr_mx_no_space, growth_scale=n_space_point
        )

        # ..............................................................................
        # Combination
        # ..............................................................................
        corr_mx = space_corr_mx * time_corr_mx
        cov_mx = np.matmul(
            np.matmul(standard_deviation_diag_mx, corr_mx), standard_deviation_diag_mx
        )

        self.corr_mx = corr_mx
        self.cov_mx = cov_mx
        return cov_mx

    def plot_measurement_space_points(self):
        """
        Visualization of the measurement points in space (x1--x2 plane).

        Intended use: quick visual verification of the input.

        Returns:
            plt: plot.
        """
        measurement_space_points = self.measurement_space_points
        groups = measurement_space_points.group
        sensor_names = measurement_space_points.sensor_name
        sensor_numbers = measurement_space_points.sensor_number
        coord_x1 = measurement_space_points.coord_x1
        coord_x2 = measurement_space_points.coord_x2

        plt.subplot()
        for group in np.unique(groups):
            idx = groups == group
            x_vec = coord_x1[idx]
            y_vec = coord_x2[idx]
            sensor_ids = [
                m + str(n) for m, n in zip(sensor_names[idx], sensor_numbers[idx])
            ]
            plt.scatter(x_vec, y_vec, label=group)
            for x, y, sensor_id in zip(x_vec, 1.01 * y_vec, sensor_ids):
                plt.text(
                    x,
                    y,
                    s=sensor_id,
                    horizontalalignment="center",
                    verticalalignment="bottom",
                )
        plt.axis("equal")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Spatial distribution of measurement points.")
        plt.legend()
        return plt

    # ==================================================================================
    # PRIVATE METHODS
    # ==================================================================================
    def _get_largest_existing_sensor_number(self, added_group, added_sensor_name):
        measurement_space_points = self.measurement_space_points

        if measurement_space_points is None:
            largest_existing_sensor_number = 0
        else:
            existing_groups = measurement_space_points.group
            existing_sensor_names = measurement_space_points.sensor_name
            existing_sensor_numbers = measurement_space_points.sensor_number

            idx_group = existing_groups == added_group
            idx_sensor_name = existing_sensor_names == added_sensor_name

            filtered_sensor_numbers = existing_sensor_numbers[
                np.logical_and(idx_group, idx_sensor_name)
            ]

            if filtered_sensor_numbers.size == 0:
                largest_existing_sensor_number = 0
            else:
                largest_existing_sensor_number = np.max(filtered_sensor_numbers)

        return largest_existing_sensor_number

    def _pre_process_space_point_input(
        self,
        coord_mx: Union[np.ndarray, Iterable, float, int],
        standard_deviation: Union[np.ndarray, Iterable, float, int],
        group: str,
        sensor_name: str,
        sensor_number: Union[np.ndarray, Iterable, float, int, None],
    ) -> np.recarray:

        # ..............................................................................
        # Preformat input
        # ..............................................................................
        if sensor_number is None:
            largest_existing_sensor_number = self._get_largest_existing_sensor_number(
                added_group=group, added_sensor_name=sensor_name
            )
            start = largest_existing_sensor_number + 1
            stop = start + np.atleast_2d(coord_mx).shape[0]
            sensor_number = np.arange(start=start, stop=stop)

        coord_mx = np.atleast_2d(coord_mx)
        standard_deviation = np.atleast_1d(standard_deviation)
        sensor_number = np.atleast_1d(sensor_number)

        n_point, n_coord_dim = coord_mx.shape
        n_standard_deviation = standard_deviation.shape[0]

        # add zero padding for missing coordinates
        if n_coord_dim < 3:
            coord_mx = np.hstack((coord_mx, np.zeros((n_point, 3 - n_coord_dim))))

        # ..............................................................................
        # Check input against existing points -> duplicates
        # ..............................................................................
        existing_measurement_space_points = self.measurement_space_points

        if existing_measurement_space_points is not None:
            existing_groups = existing_measurement_space_points.group
            existing_sensor_names = existing_measurement_space_points.sensor_name
            existing_sensor_numbers = existing_measurement_space_points.sensor_number

            idx_group = existing_groups == group
            idx_sensor_name = (
                existing_sensor_names == sensor_name
            )  # TODO, this is not gonna work for Iterable sensor_name input
            idx_sensor_number = np.in1d(existing_sensor_numbers, sensor_number)
            idx_duplicate_id = np.logical_and.reduce(
                (idx_group, idx_sensor_name, idx_sensor_number)
            )

            if np.any(idx_duplicate_id):
                duplicate_id_meas_space_points = existing_measurement_space_points[
                    idx_duplicate_id
                ]
                duplicate_id_points_table = tabulate(
                    duplicate_id_meas_space_points,
                    headers=duplicate_id_meas_space_points.dtype.names,
                    showindex=True,
                    floatfmt=".2f",
                )
                logging.warning(
                    f"{np.sum(idx_duplicate_id)} of the provided new measurement "
                    f"space points' ids (`group`, `sensor_name`, `sensor_number`) "
                    f"is(are) already present in the measurement space points."
                    f"They are these and going to be overwritten:\n "
                    f"{duplicate_id_points_table}"
                )
                # remove the duplicate id points
                self.measurement_space_points = existing_measurement_space_points[
                    np.logical_not(idx_duplicate_id)
                ]

        # ..............................................................................
        # Tile input
        # ..............................................................................
        if (n_standard_deviation == 1) and (n_point != 1):
            standard_deviation = np.tile(standard_deviation, n_point)

        group = np.tile(group, n_point)
        sensor_name = np.tile(sensor_name, n_point)

        measurement_space_points = np.core.records.fromarrays(
            arrayList=[
                group,
                sensor_name,
                sensor_number,
                standard_deviation,
                coord_mx[:, 0],
                coord_mx[:, 1],
                coord_mx[:, 2],
            ],
            names="group, sensor_name, sensor_number, standard_deviation, coord_x1, "
            "coord_x2, coord_x3",
        )
        return measurement_space_points

    def _pre_process_time_point_input(
        self, coord_vec: Union[np.ndarray, Iterable, float, int], group: str,
    ) -> np.recarray:

        coord_vec = np.atleast_1d(coord_vec)
        # ..............................................................................
        # Check input against existing points -> duplicates
        # ..............................................................................
        existing_measurement_time_points = self.measurement_time_points

        if existing_measurement_time_points is not None:
            existing_coord_t = existing_measurement_time_points.coord_t
            existing_groups = existing_measurement_time_points.group

            idx_coord_vec = existing_coord_t == coord_vec
            idx_group = existing_groups == group
            idx_duplicate_id = np.logical_and(idx_coord_vec, idx_group)

            if np.any(idx_duplicate_id):
                coord_vec = coord_vec[np.logical_not(idx_duplicate_id)]
                logging.warning(
                    f"{np.sum(idx_duplicate_id)} of the provided new measurement time "
                    f"points are already present. They are not going to be added ("
                    f"again) to the measurement time points."
                )

        n_point = coord_vec.shape[0]

        # ..............................................................................
        # Tile input
        # ..............................................................................
        group = np.tile(group, n_point)

        measurement_time_points = np.core.records.fromarrays(
            arrayList=[group, coord_vec], names="group, coord_t",
        )
        return measurement_time_points

    @staticmethod
    def _check_space_point_input_consistency(
        coord_mx: Union[np.ndarray, Iterable, float, int],
        standard_deviation: Union[np.ndarray, Iterable, float, int],
        group: str,
        sensor_name: str,
        sensor_number: Union[np.ndarray, Iterable, float, int, None],
    ) -> None:

        # Bring input shape into a desirable format: numpy arrays
        coord_mx = np.atleast_2d(coord_mx)
        standard_deviation = np.atleast_1d(standard_deviation)
        if sensor_number is None:
            sensor_number = []
        sensor_number = np.atleast_1d(sensor_number)

        n_point, n_coord_dim = coord_mx.shape
        n_standard_deviation = standard_deviation.shape[0]
        n_sensor_number = sensor_number.shape[0]

        # make some checks
        if n_coord_dim > 3:
            raise ValueError(
                f"`coord_mx` has an incorrect shape. It has {n_coord_dim} columns "
                f"while should have at most 3. The columns of `coord_mx` should "
                f"correspond to spatial coordinates."
            )

        if not isinstance(group, (str, type(None))):
            raise TypeError("If `group` is provided it should be a (single) string.")

        if not isinstance(sensor_name, (str, type(None))):
            raise TypeError(
                "If `sensor_name` is provided it should be a (single) string."
            )

        if (n_point != n_standard_deviation) and (n_standard_deviation > 1):
            raise ValueError(
                f"The number of rows of `coord_mx` ({n_point}) is not equal to the "
                f"length of `standard_deviation` ({n_standard_deviation}). The length "
                f"of `standard_deviation` should be 1 or equal to the number of rows "
                f"of `coord_mx`."
            )

        if (n_sensor_number != n_point) and (n_sensor_number > 0):
            raise ValueError(
                f"The number of elements in sensor_number ({n_sensor_number}) is not "
                f"equal to the number  rows of `coord_mx` ({n_point}). The length of "
                f"`sensor_number`should be 0 or equal to the number of rows of "
                f"`coord_mx`."
            )

    @staticmethod
    def _check_time_point_input_consistency(
        coord_vec: Union[np.ndarray, Iterable, float, int], group: str,
    ) -> None:

        # Bring input shape into a desirable format: numpy arrays
        coord_vec = np.atleast_1d(coord_vec)

        # ..............................................................................
        # Check solely the input
        # ..............................................................................
        if len(coord_vec.shape) > 1:
            raise ValueError(
                f"`coord_vec` has an incorrect shape. It has {len(coord_vec.shape)} "
                f"dimensions while should have exactly 1. `coord_vec` is intended to "
                f"express a time like variable that is univariate."
            )

        if not isinstance(group, (str, type(None))):
            raise TypeError("If `group` is provided it should be a (single) string.")
