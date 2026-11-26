"""Shared diagnostic helper for reporting calibration recall and accuracy."""

from typing import Optional, Sequence, Tuple

import numpy as np


def report_calibration_diagnostics(
    calibrator,
    res: dict,
    gt_coeffs: Optional[Sequence[float]] = None,
    gt_pairs: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    tol: float = 5.0,
) -> dict:
    """Report atlas recall, post-Hough candidate recall, and final fit accuracy.

    Parameters
    ----------
    calibrator : Calibrator
        Calibrator instance after ``fit()`` has been called.
    res : dict
        Result dictionary returned by ``calibrator.fit()``.
    gt_coeffs : sequence of float, optional
        Ground-truth polynomial coefficients (increasing power order, matching
        ``numpy.polynomial.polynomial.polyval``) mapping effective pixel to
        wavelength (Angstrom).
    gt_pairs : (pixels, wavelengths), optional
        Ground-truth (effective pixel, wavelength) pairs. If ``gt_coeffs`` is
        not given, a degree-4 polynomial is fitted to ``gt_pairs`` for
        full-detector agreement evaluation.
    tol : float, optional
        Wavelength matching tolerance in Angstrom (default: 5.0).

    Returns
    -------
    dict
        Summary metrics dictionary.
    """
    print("\n--- Calibration Diagnostics ---")
    n_peaks = len(calibrator.peaks_effective)
    n_atlas = len(calibrator.atlas_wavelengths)

    if gt_coeffs is None and gt_pairs is None:
        print(f"Detected peaks : {n_peaks}")
        print(f"Atlas lines    : {n_atlas}")
        if calibrator.candidate_peak is not None:
            print(
                f"Hough top-N    : {len(calibrator.candidate_peak)} pairs "
                f"across {len(np.unique(calibrator.candidate_peak))} peaks"
            )
        if res.get("success"):
            print(
                f"Final fit      : {len(res['matched_peaks'])} matched peaks, "
                f"RMS = {res['rms']:.3f} A"
            )
        else:
            print("Final fit      : FAILED")
        print("(No ground truth available for this example)")
        return {"success": bool(res.get("success"))}

    if gt_coeffs is not None:
        gt_coeffs = np.asarray(gt_coeffs, dtype=float)
    else:
        gt_x, gt_y = (
            np.asarray(gt_pairs[0], dtype=float),
            np.asarray(gt_pairs[1], dtype=float),
        )
        deg = min(4, len(gt_x) - 1)
        gt_coeffs = np.polynomial.polynomial.polyfit(gt_x, gt_y, deg)

    atlas_w = np.asarray(calibrator.atlas_wavelengths, dtype=float)
    peaks_eff = np.asarray(calibrator.peaks_effective, dtype=float)
    true_peak_waves = np.polynomial.polynomial.polyval(peaks_eff, gt_coeffs)

    # (1) Ground truth line recall in atlas
    # Identify which detected peaks have a matching true line in the atlas
    true_pairs_in_atlas = []
    for p_eff, tw in zip(peaks_eff, true_peak_waves):
        if len(atlas_w) == 0:
            break
        idx = int(np.argmin(np.abs(atlas_w - tw)))
        if abs(atlas_w[idx] - tw) <= tol:
            true_pairs_in_atlas.append((p_eff, atlas_w[idx]))

    n_in_atlas = len(true_pairs_in_atlas)
    atlas_recall = n_in_atlas / n_peaks if n_peaks > 0 else 0.0
    print(
        f"(1) GT recall in atlas : {n_in_atlas}/{n_peaks} detected peaks "
        f"({100.0 * atlas_recall:.1f}%, atlas size={n_atlas})"
    )

    # (2) Ground truth line recall after Hough (all candidates and top-N)
    if calibrator.candidates:
        all_hough_p = np.concatenate([cand[0] for cand in calibrator.candidates])
        all_hough_w = np.concatenate([cand[1] for cand in calibrator.candidates])
        n_recalled_hough = sum(
            1
            for p, w in true_pairs_in_atlas
            if np.any(
                np.isclose(all_hough_p, p, atol=1e-4)
                & np.isclose(all_hough_w, w, atol=1e-4)
            )
        )
    else:
        n_recalled_hough = 0

    if calibrator.candidate_peak is not None and len(calibrator.candidate_peak) > 0:
        topn_p = np.asarray(calibrator.candidate_peak, dtype=float)
        topn_w = np.asarray(calibrator.candidate_arc, dtype=float)
        n_recalled_topn = sum(
            1
            for p, w in true_pairs_in_atlas
            if np.any(
                np.isclose(topn_p, p, atol=1e-4)
                & np.isclose(topn_w, w, atol=1e-4)
            )
        )
    else:
        n_recalled_topn = 0

    hough_recall = n_recalled_hough / n_in_atlas if n_in_atlas > 0 else 0.0
    topn_recall = n_recalled_topn / n_in_atlas if n_in_atlas > 0 else 0.0
    print(
        f"(2) GT recall post-Hough: {n_recalled_hough}/{n_in_atlas} "
        f"({100.0 * hough_recall:.1f}%) in Hough candidates; "
        f"{n_recalled_topn}/{n_in_atlas} ({100.0 * topn_recall:.1f}%) in top-N"
    )

    # (3) Final recall after fitting & agreement with perfect fit
    metrics = {
        "n_peaks": n_peaks,
        "n_in_atlas": n_in_atlas,
        "atlas_recall": atlas_recall,
        "hough_recall": hough_recall,
        "topn_recall": topn_recall,
        "success": bool(res.get("success")),
    }

    if res.get("success") and res.get("fit_coeff") is not None:
        mp = np.asarray(res["matched_peaks"], dtype=float)
        ma = np.asarray(res["matched_atlas"], dtype=float)
        n_true_matched = sum(
            1
            for p, w in true_pairs_in_atlas
            if np.any(
                np.isclose(mp, p, atol=1e-4) & np.isclose(ma, w, atol=tol)
            )
        )
        final_recall = n_true_matched / n_in_atlas if n_in_atlas > 0 else 0.0
        final_precision = n_true_matched / len(mp) if len(mp) > 0 else 0.0

        pix = np.asarray(calibrator.contiguous_pixel, dtype=float)
        w_gt = np.polynomial.polynomial.polyval(pix, gt_coeffs)
        w_fit = calibrator.polyval(pix, res["fit_coeff"])
        diff = np.abs(w_fit - w_gt)
        med_err = float(np.median(diff))
        rms_err = float(np.sqrt(np.mean(diff**2)))
        max_err = float(np.max(diff))

        print(
            f"(3) Final fit recall    : {n_true_matched}/{n_in_atlas} "
            f"({100.0 * final_recall:.1f}% recall, "
            f"{n_true_matched}/{len(mp)} = {100.0 * final_precision:.1f}% precision)"
        )
        print(
            f"    Agreement vs GT fit : median={med_err:.2f} A, "
            f"RMS={rms_err:.2f} A, max={max_err:.2f} A"
        )
        metrics.update(
            {
                "n_true_matched": n_true_matched,
                "final_recall": final_recall,
                "final_precision": final_precision,
                "median_err_vs_gt": med_err,
                "rms_err_vs_gt": rms_err,
                "max_err_vs_gt": max_err,
            }
        )
    else:
        print("(3) Final fit recall    : 0/0 (calibration failed)")
        metrics.update({"n_true_matched": 0, "final_recall": 0.0})

    return metrics
