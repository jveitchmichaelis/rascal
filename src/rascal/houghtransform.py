#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hough transform

"""

import json

import numpy as np


class HoughTransform:
    """
    This handles the hough transform operations on the pixel-wavelength space.

    """

    def __init__(self):
        """
        Initialise an hough transform.

        """

        self.hough_points = np.empty(2)
        self.hough_lines = None
        self.hist = np.empty(1)
        self.xedges = np.empty(1)
        self.yedges = np.empty(1)
        self.min_slope = 1e-10
        self.max_slope = 1e10
        self.min_intercept = 1e-10
        self.max_intercept = 1e10

    def set_constraints(
        self,
        min_slope: float,
        max_slope: float,
        min_intercept: float,
        max_intercept: float,
    ):
        """
        Define the minimum and maximum of the intercepts (wavelength) and
        gradients (wavelength/pixel) that Hough pairs will be generated.

        Parameters
        ----------
        min_slope: float
            Minimum gradient for wavelength/pixel
        max_slope: float
            Maximum gradient for wavelength/pixel
        min_intercept: float
            Minimum interception point of the Hough line
        max_intercept: float
            Maximum interception point of the Hough line

        """

        assert np.isfinite(
            min_slope
        ), f"min_slope has to be finite, {min_slope} is given."
        assert np.isfinite(
            max_slope
        ), f"max_slope has to be finite, {max_slope} is given."
        assert np.isfinite(
            min_intercept
        ), f"min_intercept has to be finite, {min_intercept} is given."
        assert np.isfinite(
            max_intercept
        ), f"max_intercept has to be finite, {max_intercept} is given."

        self.min_slope = min_slope
        self.max_slope = max_slope
        self.min_intercept = min_intercept
        self.max_intercept = max_intercept

    def generate_hough_points(
        self, x: np.ndarray, y: np.ndarray, num_slopes: int
    ):
        """
        Calculate the Hough transform for a set of input points and returns the
        2D Hough hough_points matrix.

        Parameters
        ----------
        x: 1D numpy array
            The x-axis represents peaks (pixel).
        y: 1D numpy array
            The y-axis represents lines (wavelength). Vertical lines
            (i.e. infinite gradient) are not accommodated.
        num_slopes: int
            The number of slopes to be generated.

        """

        # Getting all the slopes
        slopes = np.linspace(self.min_slope, self.max_slope, num_slopes)

        # Computing all the intercepts and gradients
        intercepts = np.concatenate(y - np.outer(slopes, x))
        gradients = np.concatenate(np.column_stack([slopes] * len(x)))

        # Apply boundaries
        mask = (self.min_intercept <= intercepts) & (
            intercepts <= self.max_intercept
        )
        intercepts = intercepts[mask]
        gradients = gradients[mask]

        # Create an array of Hough Points
        self.hough_points = np.column_stack((gradients, intercepts))

    def generate_hough_points_brute_force(self, x: np.ndarray, y: np.ndarray):
        """
        Calculate the Hough transform for a set of input points and returns the
        2D Hough hough_points matrix.

        Parameters
        ----------
        x: 1D numpy array
            The x-axis represents peaks (pixel).
        y: 1D numpy array
            The y-axis represents lines (wavelength). Vertical lines
            (i.e. infinite gradient) are not accommodated.
        num_slopes: int
            The number of slopes to be generated.

        """

        idx_sort = np.argsort(x)
        x = x[idx_sort]
        y = y[idx_sort]

        gradients = []
        intercepts = []

        # For each point (x, y), computes the gradient and intercept for all
        # (X, Y) with positive gradients
        for i in range(len(x) - 1):

            gradient_tmp = (y[i + 1 :] - y[i]) / (x[i + 1 :] - x[i] + 1e-9)
            intercept_tmp = y[i + 1 :] - gradient_tmp * x[i + 1 :]

            gradients.append(gradient_tmp)
            intercepts.append(intercept_tmp)

        gradients = np.concatenate(gradients)
        intercepts = np.concatenate(intercepts)

        mask = (
            (gradients > self.min_slope)
            & (gradients < self.max_slope)
            & (intercepts > self.min_intercept)
            & (intercepts < self.max_intercept)
        )
        gradients = gradients[mask]
        intercepts = intercepts[mask]

        # Create an array of Hough Points
        self.hough_points = np.column_stack((gradients, intercepts))

    def add_hough_points(self, hough_points: np.ndarray):
        """
        Extending the Hough pairs with an externally supplied HoughTransform
        object. This can be useful if the arc lines are very concentrated in
        some wavelength ranges while nothing in available in another part.

        Parameters
        ----------
        hough_points: numpy.ndarray with 2 columns or HoughTransform object
            An externally supplied HoughTransform object that contains
            hough_points.

        """

        if isinstance(hough_points, HoughTransform):

            points = hough_points.hough_points

        elif isinstance(hough_points, np.ndarray):

            points = hough_points

        else:

            raise TypeError("Unsupported type for extending hough points.")

        self.hough_points = np.vstack((self.hough_points, points))

    def bin_hough_points(self, xbins: int, ybins: int):
        """
        Bin up data by using a 2D histogram method.

        Parameters
        ----------
        xbins: int
            The number of bins in the pixel direction.
        ybins: int
            The number of bins in the wavelength direction.

        """

        assert self.hough_points is not None, (
            "Please load an hough_points or create an hough_points with "
            "hough_points() first."
        )

        self.hist, self.xedges, self.yedges = np.histogram2d(
            self.hough_points[:, 0],
            self.hough_points[:, 1],
            bins=(xbins, ybins),
        )

        # Get the line fit_coeffients from the promising bins in the
        # histogram
        hist_sorted_arg = np.dstack(
            np.unravel_index(
                np.argsort(self.hist.ravel())[::-1], self.hist.shape
            )
        )[0]

        xbin_width = (self.xedges[1] - self.xedges[0]) / 2
        ybin_width = (self.yedges[1] - self.yedges[0]) / 2

        lines = []

        for bin_i in hist_sorted_arg:

            lines.append(
                (
                    self.xedges[bin_i[0]] + xbin_width,
                    self.yedges[bin_i[1]] + ybin_width,
                )
            )

        self.hough_lines = lines

    def save(
        self,
        filename: str = "hough_transform",
        fileformat: str = "npy",
        delimiter: str = "+",
        to_disk: bool = True,
    ):
        """
        Store the binned Hough space and/or the raw Hough pairs.

        Parameters
        ----------
        filename: str
            The filename of the output, not used if to_disk is False. It
            will be appended with the content type.
        format: str (default: 'npy')
            Choose from 'npy' and json'
        delimiter: str (default: '+')
            Delimiter for format and content types
        to_disk: boolean
            Set to True to save to disk, else return a numpy array object

        Returns
        -------
        hp_hough_points: numpy.ndarray
            only return if to_disk is False.

        """

        fileformat_split = fileformat.split(delimiter)

        if "npy" in fileformat_split:

            output_npy = []

            output_npy.append(self.hough_points)
            output_npy.append(self.hist)
            output_npy.append(self.xedges)
            output_npy.append(self.yedges)
            output_npy.append([self.min_slope])
            output_npy.append([self.max_slope])
            output_npy.append([self.min_intercept])
            output_npy.append([self.max_intercept])

            output_npy = np.array(output_npy, dtype=object)

            if to_disk:

                np.save(filename + ".npy", output_npy)

            else:

                if "json" not in fileformat_split:

                    return output_npy

        if "json" in fileformat_split:

            output_json = {}

            output_json["hough_points"] = self.hough_points.tolist()
            output_json["hist"] = self.hist.tolist()
            output_json["xedges"] = self.xedges.tolist()
            output_json["yedges"] = self.yedges.tolist()
            output_json["min_slope"] = self.min_slope
            output_json["max_slope"] = self.max_slope
            output_json["min_intercept"] = self.min_intercept
            output_json["max_intercept"] = self.max_intercept

            if to_disk:

                with open(
                    filename + ".json", "w+", encoding="ascii"
                ) as json_file:

                    json.dump(output_json, json_file)

            else:

                if "npy" in fileformat_split:

                    return output_npy, output_json

                else:

                    return output_json

    def load(self, filename="hough_transform", filetype="npy"):
        """
        Store the binned Hough space and/or the raw Hough pairs.

        Parameters
        ----------
        filename: str (default: 'hough_transform')
            The filename of the output, not used if to_disk is False. It
            will be appended with the content type.
        filetype: str (default: 'npy')
            The file type of the saved hough transform. Choose from 'npy'
            and 'json'.

        """

        if filetype == "npy":

            if filename[-4:] != ".npy":

                filename += ".npy"

            input_npy = np.load(filename, allow_pickle=True)

            self.hough_points = input_npy[0]
            self.hist = input_npy[1].astype("float")
            self.xedges = input_npy[2].astype("float")
            self.yedges = input_npy[3].astype("float")
            self.min_slope = float(input_npy[4][0])
            self.max_slope = float(input_npy[5][0])
            self.min_intercept = float(input_npy[6][0])
            self.max_intercept = float(input_npy[7][0])

        elif filetype == "json":

            if filename[-5:] != ".json":

                filename += ".json"

            input_json = json.load(open(filename, encoding="ascii"))

            self.hough_points = input_json["hough_points"]
            self.hist = np.array(input_json["hist"]).astype("float")
            self.xedges = np.array(input_json["xedges"]).astype("float")
            self.yedges = np.array(input_json["yedges"]).astype("float")
            self.min_slope = float(input_json["min_slope"])
            self.max_slope = float(input_json["max_slope"])
            self.min_intercept = float(input_json["min_intercept"])
            self.max_intercept = float(input_json["max_intercept"])

        else:

            raise ValueError(
                f"Unknown filetype {filetype}, it has to be npy or json."
            )
