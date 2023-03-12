#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Some plotting functions for diagnostic and inspection.

"""

import logging
from typing import Union

import numpy as np
from rascal import calibrator, util
from scipy import signal

logger = logging.getLogger("plotting")


def _import_matplotlib():
    """
    Call to import matplotlib.

    """

    try:

        global plt
        import matplotlib.pyplot as plt

    except ImportError:

        logger.error("matplotlib package not available.")


def _import_plotly():
    """
    Call to import plotly.

    """

    try:

        global go
        global pio
        global psp
        global pio_color
        import plotly.graph_objects as go
        import plotly.io as pio
        import plotly.subplots as psp

        pio.templates["CN"] = go.layout.Template(
            layout_colorway=[
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ]
        )

        # setting Google color palette as default
        pio.templates.default = "CN"
        pio_color = pio.templates["CN"].layout.colorway

    except ImportError:

        logger.error("plotly package not available.")


def plot_search_space(
    calibrator: "calibrator.Calibrator",
    fit_coeff: Union[list, np.ndarray] = None,
    top_n_candidate: int = 3,
    weighted: bool = True,
    save_fig: bool = False,
    fig_type: str = "png",
    filename: str = None,
    return_jsonstring: bool = False,
    renderer: str = "default",
    display: bool = True,
):
    """
    Plots the peak/arc line pairs that are considered as potential match
    candidates.

    If fit fit_coefficients are provided, the model solution will be
    overplotted.

    Parameters
    ----------
    fit_coeff: list (default: None)
        List of best polynomial fit_coefficients
    top_n_candidate: int (default: 3)
        Top ranked lines to be fitted.
    weighted: (default: True)
        Draw sample based on the distance from the matched known wavelength
        of the atlas.
    save_fig: boolean (default: False)
        Save an image if set to True. matplotlib uses the pyplot.save_fig()
        while the plotly uses the pio.write_html() or pio.write_image().
        The support format types should be provided in fig_type.
    fig_type: string (default: 'png')
        Image type to be saved, choose from:
        jpg, png, svg, pdf and iframe. Delimiter is '+'.
    filename: (default: None)
        The destination to save the image.
    return_jsonstring: (default: False)
        Set to True to save the plotly figure as json string. Ignored if
        matplotlib is used.
    renderer: (default: 'default')
        Set the rendered for the plotly display. Ignored if matplotlib is
        used.
    display: boolean (Default: False)
        Set to True to display disgnostic plot.

    Return
    ------
    json object if return_jsonstring is True.


    """

    # Get top linear estimates and combine
    candidate_peak, candidate_arc = calibrator._get_most_common_candidates(
        calibrator.candidates,
        top_n_candidate=top_n_candidate,
        weighted=weighted,
    )

    # Get the search space boundaries
    x = calibrator.contiguous_pixel

    m_1 = (
        calibrator.config.data.detector_max_wave
        - calibrator.config.data.detector_min_wave
    ) / calibrator.contiguous_pixel.max()
    y_1 = m_1 * x + calibrator.config.data.detector_min_wave

    m_2 = (
        calibrator.config.data.detector_max_wave
        + calibrator.range_tolerance
        - (
            calibrator.config.data.detector_min_wave
            + calibrator.range_tolerance
        )
    ) / calibrator.contiguous_pixel.max()
    y_2 = (
        m_2 * x
        + calibrator.config.data.detector_min_wave
        + calibrator.range_tolerance
    )

    m_3 = (
        calibrator.config.data.detector_max_wave
        - calibrator.range_tolerance
        - (
            calibrator.config.data.detector_min_wave
            - calibrator.range_tolerance
        )
    ) / calibrator.contiguous_pixel.max()
    y_3 = m_3 * x + (
        calibrator.config.data.detector_min_wave - calibrator.range_tolerance
    )

    if calibrator.plot_with_matplotlib:
        _import_matplotlib()

        fig = plt.figure(figsize=(10, 10))

        # Plot all-pairs
        plt.scatter(
            *calibrator.pairs.T, alpha=0.2, color="C0", label="All pairs"
        )

        plt.scatter(
            calibrator._merge_candidates(calibrator.candidates)[:, 0],
            calibrator._merge_candidates(calibrator.candidates)[:, 1],
            alpha=0.2,
            color="C1",
            label="Candidate Pairs",
        )

        # Tolerance region around the minimum wavelength
        plt.text(
            5,
            calibrator.config.data.detector_min_wave + 100,
            "Min wavelength (user-supplied)",
        )
        plt.hlines(
            calibrator.config.data.detector_min_wave,
            0,
            calibrator.contiguous_pixel.max(),
            color="k",
        )
        plt.hlines(
            calibrator.config.data.detector_min_wave
            + calibrator.range_tolerance,
            0,
            calibrator.contiguous_pixel.max(),
            linestyle="dashed",
            alpha=0.5,
            color="k",
        )
        plt.hlines(
            calibrator.config.data.detector_min_wave
            - calibrator.range_tolerance,
            0,
            calibrator.contiguous_pixel.max(),
            linestyle="dashed",
            alpha=0.5,
            color="k",
        )

        # Tolerance region around the maximum wavelength
        plt.text(
            5,
            calibrator.config.data.detector_max_wave + 100,
            "Max wavelength (user-supplied)",
        )
        plt.hlines(
            calibrator.config.data.detector_max_wave,
            0,
            calibrator.contiguous_pixel.max(),
            color="k",
        )
        plt.hlines(
            calibrator.config.data.detector_max_wave
            + calibrator.range_tolerance,
            0,
            calibrator.contiguous_pixel.max(),
            linestyle="dashed",
            alpha=0.5,
            color="k",
        )
        plt.hlines(
            calibrator.config.data.detector_max_wave
            - calibrator.range_tolerance,
            0,
            calibrator.contiguous_pixel.max(),
            linestyle="dashed",
            alpha=0.5,
            color="k",
        )

        # The line from (first pixel, minimum wavelength) to
        # (last pixel, maximum wavelength), and the two lines defining the
        # tolerance region.
        plt.plot(x, y_1, label="Linear Fit", color="C3")
        plt.plot(
            x, y_2, linestyle="dashed", label="Tolerance Region", color="C3"
        )
        plt.plot(x, y_3, linestyle="dashed", color="C3")

        if fit_coeff is not None:

            plt.scatter(
                calibrator.peaks,
                calibrator.polyval(calibrator.peaks, fit_coeff),
                color="C4",
                label="Solution",
            )

        plt.scatter(
            candidate_peak,
            candidate_arc,
            color="C2",
            label="Best Candidate Pairs",
        )

        plt.xlim(0, calibrator.contiguous_pixel.max())
        plt.ylim(
            calibrator.config.data.detector_min_wave
            - calibrator.range_tolerance,
            calibrator.config.data.detector_max_wave
            + calibrator.range_tolerance,
        )

        plt.ylabel("Wavelength / A")
        plt.xlabel("Pixel")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        if save_fig:

            fig_type = fig_type.split("+")

            if filename is None:

                filename_output = "rascal_hough_search_space"

            else:

                filename_output = filename

            for t in fig_type:

                if t in ["jpg", "png", "svg", "pdf"]:

                    plt.savefig(filename_output + "." + t, format=t)

        if display:

            plt.show()

        return fig

    elif calibrator.plot_with_plotly:
        _import_plotly()

        fig = go.Figure()

        # Plot all-pairs
        fig.add_trace(
            go.Scatter(
                x=calibrator.pairs[:, 0],
                y=calibrator.pairs[:, 1],
                mode="markers",
                name="All Pairs",
                marker=dict(color=pio_color[0], opacity=0.2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=calibrator._merge_candidates(calibrator.candidates)[:, 0],
                y=calibrator._merge_candidates(calibrator.candidates)[:, 1],
                mode="markers",
                name="Candidate Pairs",
                marker=dict(color=pio_color[1], opacity=0.2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=candidate_peak,
                y=candidate_arc,
                mode="markers",
                name="Best Candidate Pairs",
                marker=dict(color=pio_color[2]),
            )
        )

        # Tolerance region around the minimum wavelength
        fig.add_trace(
            go.Scatter(
                x=[0, calibrator.contiguous_pixel.max()],
                y=[
                    calibrator.config.data.detector_min_wave,
                    calibrator.config.data.detector_min_wave,
                ],
                name="Min/Maximum",
                mode="lines",
                line=dict(color="black"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, calibrator.contiguous_pixel.max()],
                y=[
                    calibrator.config.data.detector_min_wave
                    + calibrator.range_tolerance,
                    calibrator.config.data.detector_min_wave
                    + calibrator.range_tolerance,
                ],
                name="Tolerance Range",
                mode="lines",
                line=dict(color="black", dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, calibrator.contiguous_pixel.max()],
                y=[
                    calibrator.config.data.detector_min_wave
                    - calibrator.range_tolerance,
                    calibrator.config.data.detector_min_wave
                    - calibrator.range_tolerance,
                ],
                showlegend=False,
                mode="lines",
                line=dict(color="black", dash="dash"),
            )
        )

        # Tolerance region around the minimum wavelength
        fig.add_trace(
            go.Scatter(
                x=[0, calibrator.contiguous_pixel.max()],
                y=[
                    calibrator.config.data.detector_max_wave,
                    calibrator.config.data.detector_max_wave,
                ],
                showlegend=False,
                mode="lines",
                line=dict(color="black"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, calibrator.contiguous_pixel.max()],
                y=[
                    calibrator.config.data.detector_max_wave
                    + calibrator.range_tolerance,
                    calibrator.config.data.detector_max_wave
                    + calibrator.range_tolerance,
                ],
                showlegend=False,
                mode="lines",
                line=dict(color="black", dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, calibrator.contiguous_pixel.max()],
                y=[
                    calibrator.config.data.detector_max_wave
                    - calibrator.range_tolerance,
                    calibrator.config.data.detector_max_wave
                    - calibrator.range_tolerance,
                ],
                showlegend=False,
                mode="lines",
                line=dict(color="black", dash="dash"),
            )
        )

        # The line from (first pixel, minimum wavelength) to
        # (last pixel, maximum wavelength), and the two lines defining the
        # tolerance region.
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_1,
                mode="lines",
                name="Linear Fit",
                line=dict(color=pio_color[3]),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_2,
                mode="lines",
                name="Tolerance Region",
                line=dict(
                    color=pio_color[3],
                    dash="dashdot",
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_3,
                showlegend=False,
                mode="lines",
                line=dict(
                    color=pio_color[3],
                    dash="dashdot",
                ),
            )
        )

        if fit_coeff is not None:

            fig.add_trace(
                go.Scatter(
                    x=calibrator.peaks,
                    y=calibrator.polyval(calibrator.peaks, fit_coeff),
                    mode="markers",
                    name="Solution",
                    marker=dict(color=pio_color[4]),
                )
            )

        # Layout, Title, Grid config
        fig.update_layout(
            autosize=True,
            yaxis=dict(
                title="Wavelength / A",
                range=[
                    calibrator.config.data.detector_min_wave
                    - calibrator.range_tolerance * 1.1,
                    calibrator.config.data.detector_max_wave
                    + calibrator.range_tolerance * 1.1,
                ],
                showgrid=True,
            ),
            xaxis=dict(
                title="Pixel",
                zeroline=False,
                range=[0.0, calibrator.contiguous_pixel.max()],
                showgrid=True,
            ),
            hovermode="closest",
            showlegend=True,
            height=800,
            width=1000,
        )

        if save_fig:

            fig_type = fig_type.split("+")

            if filename is None:

                filename_output = "rascal_hough_search_space"

            else:

                filename_output = filename

            for t in fig_type:

                if t == "iframe":

                    pio.write_html(fig, filename_output + "." + t)

                elif t in ["jpg", "png", "svg", "pdf"]:

                    pio.write_image(fig, filename_output + "." + t)

        if display:

            if renderer == "default":

                fig.show()

            else:

                fig.show(renderer)

        if return_jsonstring:

            return fig.to_json()


def plot_fit(
    calibrator: "calibrator.Calibrator",
    fit_coeff: Union[list, np.ndarray],
    spectrum: Union[list, np.ndarray] = None,
    plot_atlas: bool = True,
    log_spectrum: bool = False,
    save_fig: bool = False,
    fig_type: str = "png",
    filename: str = None,
    return_jsonstring: bool = False,
    renderer: str = "default",
    display: bool = True,
):
    """
    Plots of the wavelength calibrated arc spectrum, the residual and the
    pixel-to-wavelength solution.

    Parameters
    ----------
    fit_coeff: 1D numpy array or list
        Best fit polynomail fit_coefficients
    spectrum: 1D numpy array (N)
        Array of length N pixels
    plot_atlas: boolean (default: True)
        Display all the relavent lines available in the atlas library.
    log_spectrum: boolean (default: False)
        Display the arc in log-space if set to True.
    save_fig: boolean (default: False)
        Save an image if set to True. matplotlib uses the pyplot.save_fig()
        while the plotly uses the pio.write_html() or pio.write_image().
        The support format types should be provided in fig_type.
    fig_type: string (default: 'png')
        Image type to be saved, choose from:
        jpg, png, svg, pdf and iframe. Delimiter is '+'.
    filename: string (default: None)
        Provide a filename or full path. If the extension is not provided
        it is defaulted to png.
    return_jsonstring: boolean (default: False)
        Set to True to return json strings if using plotly as the plotting
        library.
    renderer: string (default: 'default')
        Indicate the Plotly renderer. Nothing gets displayed if
        return_jsonstring is set to True.
    display: boolean (Default: False)
        Set to True to display disgnostic plot.

    Returns
    -------
    Return json strings if using plotly as the plotting library and json
    is True.

    """

    if spectrum is None:

        try:

            spectrum = calibrator.spectrum

        except Exception as e:

            calibrator.logger.error(e)
            calibrator.logger.error(
                "Spectrum is not provided, it cannot be plotted."
            )

    if spectrum is not None:

        if log_spectrum:

            spectrum[spectrum < 0] = 1e-100
            spectrum = np.log10(spectrum)
            vline_max = np.nanmax(spectrum) * 2.0
            text_box_pos = 1.2 * max(spectrum)

        else:

            vline_max = np.nanmax(spectrum) * 1.2
            text_box_pos = 0.8 * max(spectrum)

    else:

        vline_max = 1.0
        text_box_pos = 0.5

    wave = calibrator.polyval(calibrator.contiguous_pixel, fit_coeff)

    fitted_diff = []

    for p, x in zip(calibrator.matched_peaks, calibrator.matched_atlas):

        diff = calibrator.atlas_wavelengths - calibrator.polyval(p, fit_coeff)
        idx = np.argmin(np.abs(diff))

        calibrator.logger.info(f"Peak at: {x} A")

        fitted_diff.append(diff[idx])
        calibrator.logger.info(
            f"- matched to {calibrator.atlas_wavelengths[idx]} A"
        )

    if calibrator.plot_with_matplotlib:

        _import_matplotlib()

        fig, (ax1, ax2, ax3) = plt.subplots(
            nrows=3, sharex=True, gridspec_kw={"hspace": 0.0}, figsize=(15, 9)
        )
        fig.tight_layout()

        # Plot fitted spectrum
        if spectrum is not None:

            ax1.plot(wave, spectrum, label="Arc Spectrum")
            ax1.vlines(
                calibrator.polyval(calibrator.peaks_effective, fit_coeff),
                np.array(spectrum)[calibrator.peaks.astype("int")],
                vline_max,
                linestyles="dashed",
                colors="C1",
                label="Detected Peaks",
            )

        # Plot the atlas
        if plot_atlas:

            # spec = SyntheticSpectrum(
            #    fit, model_type='poly', degree=len(fit)-1)
            # x_locs = spec.get_pixels(calibrator.atlas)
            ax1.vlines(
                calibrator.atlas_wavelengths,
                0,
                vline_max,
                colors="C2",
                label="Given Lines",
            )

        first_one = True
        for p, x in zip(calibrator.matched_peaks, calibrator.matched_atlas):

            p_idx = int(
                calibrator.peaks[np.where(calibrator.peaks_effective == p)[0]]
            )

            if spectrum is not None:

                if first_one:
                    ax1.vlines(
                        calibrator.polyval(p, fit_coeff),
                        spectrum[p_idx],
                        vline_max,
                        colors="C1",
                        label="Fitted Peaks",
                    )
                    first_one = False

                else:
                    ax1.vlines(
                        calibrator.polyval(p, fit_coeff),
                        spectrum[p_idx],
                        vline_max,
                        colors="C1",
                    )

            ax1.text(
                x - 3,
                text_box_pos,
                s=(
                    f"{calibrator.atlas_lines[idx].element}:"
                    + f"{calibrator.atlas_lines[idx].wavelength:1.2f}"
                ),
                rotation=90,
                bbox=dict(facecolor="white", alpha=1),
            )

        rms = np.sqrt(np.mean(np.array(fitted_diff) ** 2.0))

        ax1.grid(linestyle=":")
        ax1.set_ylabel("Electron Count / e-")

        if spectrum is not None:

            if log_spectrum:

                ax1.set_ylim(0, vline_max)

            else:

                ax1.set_ylim(np.nanmin(spectrum), vline_max)

        ax1.legend(loc="center right")

        # Plot the residuals
        ax2.scatter(
            calibrator.polyval(calibrator.matched_peaks, fit_coeff),
            fitted_diff,
            marker="+",
            color="C1",
        )
        ax2.hlines(0, wave.min(), wave.max(), linestyles="dashed")
        ax2.hlines(
            rms,
            wave.min(),
            wave.max(),
            linestyles="dashed",
            color="k",
            label="RMS",
        )
        ax2.hlines(
            -rms, wave.min(), wave.max(), linestyles="dashed", color="k"
        )
        ax2.grid(linestyle=":")
        ax2.set_ylabel("Residual / A")
        ax2.legend()

        # Plot the polynomial
        ax3.scatter(
            calibrator.polyval(calibrator.matched_peaks, fit_coeff),
            calibrator.matched_peaks,
            marker="+",
            color="C1",
            label="Fitted Peaks",
        )
        ax3.plot(
            wave, calibrator.contiguous_pixel, color="C2", label="Solution"
        )
        ax3.grid(linestyle=":")
        ax3.set_xlabel("Wavelength / A")
        ax3.set_ylabel("Pixel")
        ax3.legend(loc="lower right")
        w_min = calibrator.polyval(min(calibrator.matched_peaks), fit_coeff)
        w_max = calibrator.polyval(max(calibrator.matched_peaks), fit_coeff)
        ax3.set_xlim(w_min * 0.95, w_max * 1.05)

        plt.tight_layout()

        if save_fig:

            fig_type = fig_type.split("+")

            if filename is None:

                filename_output = "rascal_solution"

            else:

                filename_output = filename

            for t in fig_type:

                if t in ["jpg", "png", "svg", "pdf"]:

                    plt.savefig(filename_output + "." + t, format=t)

        if display:

            fig.show()

        return fig

    elif calibrator.plot_with_plotly:

        _import_plotly()

        fig = go.Figure()

        # Top plot - arc spectrum and matched peaks
        if spectrum is not None:
            fig.add_trace(
                go.Scatter(
                    x=wave,
                    y=spectrum,
                    mode="lines",
                    yaxis="y3",
                    name="Arc Spectrum",
                )
            )

            spec_max = np.nanmax(spectrum) * 1.05

        else:

            spec_max = vline_max

        y_fitted = []

        for p in calibrator.peaks_effective:

            x = calibrator.polyval(p, fit_coeff)

            p_idx = int(
                calibrator.peaks[np.where(calibrator.peaks_effective == p)[0]]
            )

            # Add vlines
            fig.add_shape(
                type="line",
                xref="x",
                yref="y3",
                x0=x,
                y0=spectrum[p_idx],
                x1=x,
                y1=spec_max,
                line=dict(color=pio_color[1], width=1),
            )

            if p in calibrator.matched_peaks:

                y_fitted.append(spectrum[p_idx])

        x_fitted = calibrator.polyval(calibrator.matched_peaks, fit_coeff)

        fig.add_trace(
            go.Scatter(
                x=x_fitted,
                y=y_fitted,
                mode="markers",
                marker=dict(color=pio_color[1]),
                yaxis="y3",
                showlegend=False,
            )
        )

        # Middle plot - Residual plot
        rms = np.sqrt(np.mean(np.array(fitted_diff) ** 2.0))
        fig.add_trace(
            go.Scatter(
                x=x_fitted,
                y=fitted_diff,
                mode="markers",
                marker=dict(color=pio_color[1]),
                yaxis="y2",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[wave.min(), wave.max()],
                y=[0, 0],
                mode="lines",
                line=dict(color=pio_color[0], dash="dash"),
                yaxis="y2",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[wave.min(), wave.max()],
                y=[rms, rms],
                mode="lines",
                line=dict(color="black", dash="dash"),
                yaxis="y2",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[wave.min(), wave.max()],
                y=[-rms, -rms],
                mode="lines",
                line=dict(color="black", dash="dash"),
                yaxis="y2",
                name="RMS",
            )
        )

        # Bottom plot - Polynomial fit for Pixel to Wavelength
        fig.add_trace(
            go.Scatter(
                x=x_fitted,
                y=calibrator.matched_peaks,
                mode="markers",
                marker=dict(color=pio_color[1]),
                yaxis="y1",
                name="Fitted Peaks",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=wave,
                y=calibrator.contiguous_pixel,
                mode="lines",
                line=dict(color=pio_color[2]),
                yaxis="y1",
                name="Solution",
            )
        )

        # Layout, Title, Grid config
        if spectrum is not None:

            if log_spectrum:

                fig.update_layout(
                    yaxis3=dict(
                        title="Electron Count / e-",
                        range=[
                            np.log10(np.percentile(spectrum, 15)),
                            np.log10(spec_max),
                        ],
                        domain=[0.67, 1.0],
                        showgrid=True,
                        type="log",
                    )
                )

            else:

                fig.update_layout(
                    yaxis3=dict(
                        title="Electron Count / e-",
                        range=[np.percentile(spectrum, 15), spec_max],
                        domain=[0.67, 1.0],
                        showgrid=True,
                    )
                )

        fig.update_layout(
            autosize=True,
            yaxis2=dict(
                title="Residual / A",
                range=[min(fitted_diff), max(fitted_diff)],
                domain=[0.33, 0.66],
                showgrid=True,
            ),
            yaxis=dict(
                title="Pixel",
                range=[0.0, max(calibrator.contiguous_pixel)],
                domain=[0.0, 0.32],
                showgrid=True,
            ),
            xaxis=dict(
                title="Wavelength / A",
                zeroline=False,
                range=[
                    calibrator.polyval(
                        min(calibrator.matched_peaks), fit_coeff
                    )
                    * 0.95,
                    calibrator.polyval(
                        max(calibrator.matched_peaks), fit_coeff
                    )
                    * 1.05,
                ],
                showgrid=True,
            ),
            hovermode="closest",
            showlegend=True,
            height=800,
            width=1000,
        )

        if save_fig:

            fig_type = fig_type.split("+")

            if filename is None:

                filename_output = "rascal_solution"

            else:

                filename_output = filename

            for t in fig_type:

                if t == "iframe":

                    pio.write_html(fig, filename_output + "." + t)

                elif t in ["jpg", "png", "svg", "pdf"]:

                    pio.write_image(fig, filename_output + "." + t)

        if display:

            if renderer == "default":

                fig.show()

            else:

                fig.show(renderer)

        if return_jsonstring:

            return fig.to_json()

    else:

        assert (
            calibrator.matplotlib_imported
        ), "matplotlib package not available. Plot cannot be generated."
        assert (
            calibrator.plotly_imported
        ), "plotly package is not available. Plot cannot be generated."


def plot_arc(
    calibrator: "calibrator.Calibrator",
    contiguous_pixel: Union[list, np.ndarray] = None,
    log_spectrum: Union[list, np.ndarray] = False,
    save_fig: bool = False,
    fig_type: str = "png",
    filename: str = None,
    return_jsonstring: bool = False,
    renderer: str = "default",
    display: bool = True,
):
    """
    Plots the 1D spectrum of the extracted arc.

    parameters
    ----------
    contiguous_pixel: array (default: None)
        pixel value of the of the spectrum, this is only needed if the
        spectrum spans multiple detector arrays.
    log_spectrum: boolean (default: False)
        Set to true to display the wavelength calibrated arc spectrum in
        logarithmic space.
    save_fig: boolean (default: False)
        Save an image if set to True. matplotlib uses the pyplot.save_fig()
        while the plotly uses the pio.write_html() or pio.write_image().
        The support format types should be provided in fig_type.
    fig_type: string (default: 'png')
        Image type to be saved, choose from:
        jpg, png, svg, pdf and iframe. Delimiter is '+'.
    filename: string (default: None)
        Provide a filename or full path. If the extension is not provided
        it is defaulted to png.
    return_jsonstring: boolean (default: False)
        Set to True to return json strings if using plotly as the plotting
        library.
    renderer: string (default: 'default')
        Indicate the Plotly renderer. Nothing gets displayed if
        return_jsonstring is set to True.
    display: boolean (Default: False)
        Set to True to display disgnostic plot.

    Returns
    -------
    Return json strings if using plotly as the plotting library and json
    is True.

    """

    if contiguous_pixel is None:

        contiguous_pixel = calibrator.contiguous_pixel

    if calibrator.plot_with_matplotlib:

        _import_matplotlib()

        fig = plt.figure(figsize=(18, 5))

        if calibrator.spectrum is not None:
            if log_spectrum:
                plt.plot(
                    contiguous_pixel,
                    np.log10(calibrator.spectrum / calibrator.spectrum.max()),
                    label="Arc Spectrum",
                )
                plt.vlines(
                    calibrator.peaks_effective,
                    -2,
                    0,
                    label="Detected Peaks",
                    color="C1",
                )
                plt.ylabel("log(Normalised Count)")
                plt.ylim(-2, 0)
            else:
                plt.plot(
                    contiguous_pixel,
                    calibrator.spectrum / calibrator.spectrum.max(),
                    label="Arc Spectrum",
                )
                plt.ylabel("Normalised Count")
                plt.vlines(
                    calibrator.peaks_effective,
                    0,
                    1.05,
                    label="Detected Peaks",
                    color="C1",
                )
            plt.title("Number of pixels: " + str(calibrator.spectrum.shape[0]))
            plt.xlim(0, calibrator.spectrum.shape[0])
            plt.legend()

        else:

            plt.xlim(0, max(calibrator.peaks_effective))

        plt.xlabel("Pixel (Spectral Direction)")
        plt.grid()
        plt.tight_layout()

        if save_fig:

            fig_type = fig_type.split("+")

            if filename is None:

                filename_output = "rascal_arc"

            else:

                filename_output = filename

            for t in fig_type:

                if t in ["jpg", "png", "svg", "pdf"]:

                    plt.savefig(filename_output + "." + t, format=t)

        if display:

            plt.show()

        return fig

    if calibrator.plot_with_plotly:

        _import_plotly()

        fig = go.Figure()

        if log_spectrum:

            # Plot all-pairs
            fig.add_trace(
                go.Scatter(
                    x=list(contiguous_pixel),
                    y=list(
                        np.log10(
                            calibrator.spectrum / calibrator.spectrum.max()
                        )
                    ),
                    mode="lines",
                    name="Arc",
                )
            )
            xmin = min(
                np.log10(calibrator.spectrum / calibrator.spectrum.max())
            )
            xmax = max(
                np.log10(calibrator.spectrum / calibrator.spectrum.max())
            )

        else:

            # Plot all-pairs
            fig.add_trace(
                go.Scatter(
                    x=list(contiguous_pixel),
                    y=list(calibrator.spectrum / calibrator.spectrum.max()),
                    mode="lines",
                    name="Arc",
                )
            )
            xmin = min(calibrator.spectrum / calibrator.spectrum.max())
            xmax = max(calibrator.spectrum / calibrator.spectrum.max())

        # Add vlines
        for i in calibrator.peaks_effective:
            fig.add_shape(
                type="line",
                xref="x",
                yref="y",
                x0=i,
                y0=0,
                x1=i,
                y1=1.05,
                line=dict(color=pio_color[1], width=1),
            )

        fig.update_layout(
            autosize=True,
            yaxis=dict(
                title="Normalised Count", range=[xmin, xmax], showgrid=True
            ),
            xaxis=dict(
                title="Pixel",
                zeroline=False,
                range=[0.0, len(calibrator.spectrum)],
                showgrid=True,
            ),
            hovermode="closest",
            showlegend=True,
            height=800,
            width=1000,
        )

        fig.update_xaxes(
            showline=True, linewidth=1, linecolor="black", mirror=True
        )

        fig.update_yaxes(
            showline=True, linewidth=1, linecolor="black", mirror=True
        )

        if save_fig:

            fig_type = fig_type.split("+")

            if filename is None:

                filename_output = "rascal_arc"

            else:

                filename_output = filename

            for t in fig_type:

                if t == "iframe":

                    pio.write_html(fig, filename_output + "." + t)

                elif t in ["jpg", "png", "svg", "pdf"]:

                    pio.write_image(fig, filename_output + "." + t)

        if display:

            if renderer == "default":

                fig.show()

            else:

                fig.show(renderer)

        if return_jsonstring:

            return fig.to_json()


def plot_calibration_lines(
    elements: Union[list, np.ndarray] = [],
    linelist: str = "nist",
    min_atlas_wavelength: float = 3000.0,
    max_atlas_wavelength: float = 15000.0,
    min_intensity: float = 5.0,
    min_distance: float = 0.0,
    brightest_n_lines: int = None,
    pixel_scale: float = 1.0,
    vacuum: bool = False,
    pressure: float = 101325.0,
    temperature: float = 273.15,
    relative_humidity: float = 0.0,
    label: bool = False,
    log: bool = False,
    save_fig: bool = False,
    fig_type: str = "png",
    filename: str = None,
    display: bool = True,
    fig_kwarg: dict = {"figsize": (12, 8)},
):
    """
    Plot the expected arc spectrum. Currently only available with matplotlib.

    Parameters
    ----------
    elements: list
        List of short element names, e.g. He as per NIST
    linelist: str
        Either 'nist' to use the default lines or path to a linelist file.
    min_atlas_wavelength: int
        Minimum wavelength to search, Angstrom
    max_atlas_wavelength: int
        Maximum wavelength to search, Angstrom
    min_intensity: int
        Minimum intensity to search, per NIST
    min_distance: int
        All ines within this distance from other lines are treated
        as unresolved, all of them get removed from the list.
    brightest_n_lines: int
        Only return the n brightest lines
    vacuum: bool
        Return vacuum wavelengths
    pressure: float
        Atmospheric pressure, Pascal
    temperature: float
        Temperature in Kelvin, default room temp
    relative_humidity: float
        Relative humidity, percent
    log: bool
        Plot intensities in log scale
    save_fig: boolean (default: False)
        Save an image if set to True. matplotlib uses the pyplot.save_fig()
        while the plotly uses the pio.write_html() or pio.write_image().
        The support format types should be provided in fig_type.
    fig_type: string (default: 'png')
        Image type to be saved, choose from:
        jpg, png, svg, pdf and iframe. Delimiter is '+'.
    filename: string (default: None)
        Provide a filename or full path. If the extension is not provided
        it is defaulted to png.
    display: boolean (Default: False)
        Set to True to display disgnostic plot.

    Returns
    -------
    fig: matplotlib figure object

    """

    _import_matplotlib()

    # the min_intensity and min_distance are set to 0.0 because the
    # simulated spectrum would contain them. These arguments only
    # affect the labelling.
    (
        element_list,
        wavelength_list,
        intensity_list,
    ) = util.load_calibration_lines(
        elements=elements,
        linelist=linelist,
        min_atlas_wavelength=min_atlas_wavelength,
        max_atlas_wavelength=max_atlas_wavelength,
        min_intensity=0.0,
        min_distance=0.0,
        brightest_n_lines=brightest_n_lines,
        vacuum=vacuum,
        pressure=pressure,
        temperature=temperature,
        relative_humidity=relative_humidity,
    )

    # Nyquist sampling rate (2.5) for CCD at seeing of 1 arcsec
    sigma = pixel_scale * 2.5 * 1.0
    x = np.arange(-100, 100.001, 0.001)
    gaussian = util.gauss(x, a=1.0, x0=0.0, sigma=sigma)

    # Generate the equally spaced-wavelength array, and the
    # corresponding intensity
    w = np.around(
        np.arange(min_atlas_wavelength, max_atlas_wavelength + 0.001, 0.001),
        decimals=3,
    ).astype("float64")
    i = np.zeros_like(w)

    for e in elements:
        i[
            np.isin(
                w, np.around(wavelength_list[element_list == e], decimals=3)
            )
        ] += intensity_list[element_list == e]
    # Convolve to simulate the arc spectrum
    model_spectrum = signal.convolve(i, gaussian, mode="same")

    # now clean up by min_intensity and min_distance
    intensity_mask = util.filter_intensity(
        elements,
        np.column_stack((element_list, wavelength_list, intensity_list)),
        min_intensity=min_intensity,
    )
    wavelength_list = wavelength_list[intensity_mask]
    intensity_list = intensity_list[intensity_mask]
    element_list = element_list[intensity_mask]

    distance_mask = util.filter_distance(
        wavelength_list, min_distance=min_distance
    )
    wavelength_list = wavelength_list[distance_mask]
    intensity_list = intensity_list[distance_mask]
    element_list = element_list[distance_mask]

    fig = plt.figure(**fig_kwarg)

    for j, e in enumerate(elements):
        e_mask = element_list == e
        markerline, stemline, _ = plt.stem(
            wavelength_list[e_mask],
            intensity_list[e_mask],
            label=e,
            linefmt=f"C{j}-",
        )
        plt.setp(stemline, linewidth=2.0)
        plt.setp(markerline, markersize=2.5, color=f"C{j}")

        if label:

            for _w in wavelength_list[e_mask]:

                plt.text(
                    _w,
                    max(model_spectrum) * 1.05,
                    s=f"{e}: {_w:1.2f}",
                    rotation=90,
                    bbox=dict(facecolor="white", alpha=1),
                )

            plt.vlines(
                wavelength_list[e_mask],
                intensity_list[e_mask],
                max(model_spectrum) * 1.25,
                linestyles="dashed",
                lw=0.5,
                color="grey",
            )

    plt.plot(w, model_spectrum, lw=1.0, c="k", label="Simulated Arc Spectrum")
    if vacuum:
        plt.xlabel("Vacuum Wavelength / A")
    else:
        plt.xlabel("Air Wavelength / A")
    plt.ylabel("NIST intensity")
    plt.grid()
    plt.xlim(min(w), max(w))
    plt.ylim(0, max(model_spectrum) * 1.25)
    plt.legend()
    plt.tight_layout()
    if log:
        plt.ylim(ymin=min_intensity * 0.75)
        plt.yscale("log")

    if save_fig:

        fig_type = fig_type.split("+")

        if filename is None:

            filename_output = "rascal_arc"

        else:

            filename_output = filename

        for t in fig_type:

            if t in ["jpg", "png", "svg", "pdf"]:

                plt.savefig(filename_output + "." + t, format=t)

    if display:

        plt.show()

    return fig
