"""
Test to ensure that the covariance and correlation matrices produced by
MeasurementSpaceTimePoints when passing the correlation as a kernel class
(test) and as a callable (reference) are equal.
"""

import numpy as np
from scipy import stats

from tripy.base import MeasurementSpaceTimePoints
from tripy.kernels import Exponential
from tripy.utils import correlation_function


def test_MeasurementSpaceTimePoints():

    # Input data
    intertia = 1e-3  # second moment of inertia in m**4
    L = 10  # span in m
    Q = 1e5  # point load in N

    # Physical model
    def beam_deflection(load, span, inertia, elastic_mod, a, d):
        """
        Computes the deflection of a simply supported beam under a point load.
        Args:
            load: point load intensity.
            span: length of the beam.
            inertia: second moment of inertia.
            elastic_mod: Young`s modulus
            a: distance between the point load and the further extreme of
            the beam called "O".
            d: distance between the point where the displacement is calculated
             and "O"

        Returns:
            x_predicted: displacement.
        """
        b = span - a
        if a < d:
            raise Exception("a should be bigger-equal than x")
        x_predicted = (
            load
            * b
            * d
            / (6 * 10**9 * span * elastic_mod * inertia)
            * (span**2 - b**2 - d**2)
        )
        return x_predicted

    def physical_model_fun(theta):
        x1 = beam_deflection(
            load=Q, span=L, inertia=intertia, elastic_mod=theta, a=L / 2, d=L / 2
        )
        x2 = beam_deflection(
            load=Q, span=L, inertia=intertia, elastic_mod=theta, a=L / 2, d=L / 4
        )
        x3 = beam_deflection(
            load=Q, span=L, inertia=intertia, elastic_mod=theta, a=3 * L / 4, d=L / 2
        )
        x4 = beam_deflection(
            load=Q,
            span=L,
            inertia=intertia,
            elastic_mod=theta,
            a=3 * L / 4,
            d=3 * L / 4,
        )
        x_model = np.array([[x1, x2], [x3, x4]])
        return x_model

    # log-prior
    mu_0 = 30  # GPa
    sd_0 = 30  # GPa

    def log_prior(theta):
        return stats.norm.logpdf(theta, mu_0, sd_0)

    # Adding measurement points
    MS_test = MeasurementSpaceTimePoints()
    MS_ref = MeasurementSpaceTimePoints()

    MS_test.add_measurement_space_points(
        coord_mx=[[L / 2, 1], [L / 4, 1]],
        group="translation",
        sensor_name="TNO",
        standard_deviation=0.01,
    )

    MS_test.add_measurement_time_points(
        coord_vec=[1, 3.5],
        group="truckload",
    )

    MS_ref.add_measurement_space_points(
        coord_mx=[[L / 2, 1], [L / 4, 1]],
        group="translation",
        sensor_name="TNO",
        standard_deviation=0.01,
    )

    MS_ref.add_measurement_time_points(
        coord_vec=[1, 3.5],
        group="truckload",
    )

    # Add correlation function
    def correlation_func_translation(d):
        return correlation_function(d, correlation_length=2.0)

    def correlation_func_truck_load(d):
        return correlation_function(d, correlation_length=3.0)

    MS_test.add_measurement_space_within_group_correlation(
        group="translation", correlation_func=Exponential, length_scale=2.0
    )
    MS_test.add_measurement_time_within_group_correlation(
        group="truckload", correlation_func=Exponential, length_scale=3.0
    )

    MS_ref.add_measurement_space_within_group_correlation(
        group="translation", correlation_func=correlation_func_translation
    )
    MS_ref.add_measurement_time_within_group_correlation(
        group="truckload", correlation_func=correlation_func_truck_load
    )

    # Get test and reference covariance matrices
    cov_mx_test = MS_test.compile_covariance_matrix()
    cov_mx_ref = MS_ref.compile_covariance_matrix()

    # Get test and reference correlation matrices
    corr_mx_test = MS_test.corr_mx
    corr_mx_ref = MS_ref.corr_mx

    # Assert equal
    assert np.allclose(cov_mx_test, cov_mx_ref)
    assert np.allclose(corr_mx_test, corr_mx_ref)
