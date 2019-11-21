# -*- coding: utf-8 -*-

"""Sentinel 1 inversion class"""
import logging

from pathlib import Path

import datetime as dt
import numpy as np

from scipy.interpolate import interp1d

from .s1_observations import Sentinel1Observations

from .utils import define_temporal_grid

from .watercloudmodel import cost_function

from .kaska import Sentinel2Data

import scipy.optimize

from skimage.filters import sobel

from skimage.segmentation import watershed

LOG = logging.getLogger(__name__)


class KaSKASAR(object):
    """A class to process Sentinel 1 SAR data using S2 data as 
    an input"""

    def __init__(
        self, time_grid, state_mask, s1_observations, s2_data, prior, chunk=None
    ):
        """Set up processing paths and options for s1 observations
        
        Parameters
        ----------
        time_grid : iter
            A temporal grid. E.g. a list of datetimes
        state_mask : str
            The state mask file. Must be readable by GDAL and be georeferenced
        s1_observations : s1_observations
            S1 observations type
        s2_data : s2_observations
            S2 observations type
        prior : dict
            The prior distribution
        chunk : inteter, optional
            The chunk if processing by chunks. Used for file outputs, by default None
        """
        self.time_grid = time_grid
        self.s1_observations = s1_observations
        self.state_mask = state_mask
        self.output_folder = Path(output_folder) / ("S1_outputs")
        self.chunk = chunk
        self.s2_data = self._resampling_times(s2_data)

    def _resampling_times(self, s2_data):
        """Resample S2 smoothed output to match S1 observations
        times"""
        # Move everything to DoY to simplify interpolation
        s2_doys = [
            int(dt.datetime.strftime(x, "%j")) for x in s2_data.temporal_grid
        ]
        s1_doys = [
            int(dt.datetime.strftime(x, "%j"))
            for x in self.s1_observations.dates.keys()
        ]
        n_sar_obs = len(s1_doys)
        # Interpolate S2 retrievals to S1 time grid
        f = interp1d(s2_doys, s2_data.slai, axis=0, bounds_error=False)
        lai_s1 = f(s1_doys)
        f = interp1d(s2_doys, s2_data.scab, axis=0, bounds_error=False)
        cab_s1 = f(s1_doys)
        f = interp1d(s2_doys, s2_data.scbrown, axis=0, bounds_error=False)
        cbrown_s1 = f(s1_doys)
        return Sentinel2Data(lai_s1, cab_s1, cbrown_s1)


    def _segment(self, lai, markers=250, compactness=0.001):
        L = lai.max(axis=0)  # I think
        gradient = sobel(L)
        patches = watershed(gradient, markers=markers,
                            compactness=compactness)

    def sentinel1_inversion(self, segment=False):
        nt, ny, nx = self.s2_data.slai.shape

        # FY Model is slightly different, boundaries & parameter names need to
        # be different
        outputs = {
            param: np.zeros((nt, ny, nx))
            for param in ["Avv", "Bvv", "Cvv" "Avh", "Bvh", "Cvh"]
        }
        bounds = [
            [-40, -5],
            [1e-4, 1],
            [-40, -1],
            [-40, -5],
            [1e-4, 1],
            [-40, -1],
            *([[0.01, 1]] * nt),
        ]
        # FY Now process pixel by pixel
        # Or segment and process patch by patch
        # see self._segment
        for (row, col) in np.ndindex(*self.s2_data.slai[0].shape):

            lai = self.s2_data.slai[:, row, col]
            svv = 10 * np.log10(self.s1_observations.VV[:, row, col])
            svh = 10 * np.log10(self.s1_observations.VH[:, row, col])
            theta = self.s1_observations.theta[:, row, col]
            ## FY Need to extract prior distribution from prior object
            prior_mean = 1.0
            prior_unc = 1.0 # placeholder
            ### FY GDAL doesn't get the netCDF metadata with the orbits
            ### But basically, one would need a loop over individual orbits here
            # orbits = get_orbit_no()
            # for orbit in orbits:
            # Now minimise for this pixel:
            # set x0 to prior value
            # FY set gamma to a constant? -> check sar_homework repo for 
            # value we used in notebooks
            gamma = (123, 456)  # FY Placeholder
            x0 = prior_mean  # FY Placeholder
            # FY, the next commented block is copied from the notebook.
            # retval is the result of the minimisation, the residuals are not
            # used, we can probably fix the soil parameters here, or
            # (1.99, 38.9 etc) or fish them out from the prior object
            ###############################################################
            # # # #    alpha_mean = fresnel(mv2eps(1.99, 38.9, 11.5, sm_mean), theta.mean())
            # # # # alpha_std = np.ones_like(alpha_mean)*0.2
            # # # # prior_mean = np.concatenate([[0,]*6, alpha_mean, np.zeros_like(sm_mean), s2_lai])
            # # # # prior_sd = np.concatenate([[10., ]*6, alpha_std, [0.1,]*n_obs, [0.05, ]*n_obs])
            # # # # gamma = (10000, 500)
            # # # # retval, residuals = invert_field_(svv, svh, theta, prior_mean, prior_sd, gamma, s2_lai)
            # # # # alpha = retval.x[6:(6+len(svv))]
            # # # # sols = quad_approx_solver(1.99, 38.9, 11.5,theta, alpha)
            ###############################################################

            retval = scipy.optimize.minimize(
                cost_function, x0, args=(svh, svv, theta, gamma, prior_mean, prior_unc)
            )
            # FY store retrieved parameters
            for i, raster in enumerate(outputs):
                raster[row, col] = retval.x[i]
    # Resample to time grid
    # Return and save there, or add method in class to dump to disk
    # 