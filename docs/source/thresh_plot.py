import matplotlib.pyplot as plt
import numpy as np

plt.figure(1)
plt.clf()
# data point
pix = np.arange(0, 1001, 50)
wave = 4000.0 + pix**0.63437 * 50.0
plt.scatter(
    pix, wave, label="Solution", c="midnightblue", s=10, marker="x", zorder=15
)

# minmax & range tolerance
wave_min = 4000.0
wave_max = 8000.0
range_tolerance = 500.0
plt.plot(
    [pix[0], pix[-1]],
    [wave_min - range_tolerance, wave_min - range_tolerance],
    ls=":",
    lw=1,
    c="orangered",
    label="",
)
plt.plot(
    [pix[0], pix[-1]],
    [wave_min + range_tolerance, wave_min + range_tolerance],
    ls=":",
    lw=1,
    c="orangered",
)
plt.plot(
    [pix[0], pix[-1]], [wave_min, wave_min], c="orangered", lw=1, label=""
)
plt.plot(
    [pix[0], pix[-1]],
    [wave_max - range_tolerance, wave_max - range_tolerance],
    ls=":",
    lw=1,
    c="orangered",
    label="",
)
plt.plot(
    [pix[0], pix[-1]],
    [wave_max + range_tolerance, wave_max + range_tolerance],
    ls=":",
    lw=1,
    c="orangered",
)
plt.plot(
    [pix[0], pix[-1]], [wave_max, wave_max], c="orangered", lw=1, label=""
)

plt.annotate(
    s="",
    xy=(100.0, 8050),
    xytext=(100.0, 7450),
    arrowprops=dict(arrowstyle="<->", color="orangered"),
)
plt.text(115.0, 7650.0, s="range_tolerance", c="orangered")

plt.text(650, 3800, s="min_wavelength = 4000", c="orangered")
plt.text(10, 8100, s="max_wavelength = 8000", c="orangered")

# linear_tolerance
linear_tolerance = 250.0
plt.plot([pix[0], pix[-1]], [wave_min, wave_max], c="green", label="")

plt.plot(
    [pix[0], pix[-1]],
    [wave_min - range_tolerance, wave_max - range_tolerance],
    ls="--",
    lw=1,
    c="green",
    label="",
)
plt.plot(
    [pix[0], pix[-1]],
    [
        wave_min - range_tolerance - linear_tolerance,
        wave_max - range_tolerance - linear_tolerance,
    ],
    ls="--",
    lw=1,
    c="green",
    label="",
)

plt.plot(
    [pix[0], pix[-1]],
    [wave_min + range_tolerance, wave_max + range_tolerance],
    ls="--",
    lw=1,
    c="green",
    label="",
)
plt.plot(
    [pix[0], pix[-1]],
    [
        wave_min + range_tolerance + linear_tolerance,
        wave_max + range_tolerance + linear_tolerance,
    ],
    ls="--",
    lw=1,
    c="green",
    label="",
)

plt.annotate(
    s="",
    xy=(100.0, 4820),
    xytext=(100.0, 5220),
    arrowprops=dict(arrowstyle="<->", color="green"),
)
plt.text(20.0, 5050.0, s="linearity_thresh", c="green", rotation=26.0)

# solution space
plt.gca().add_patch(
    plt.Polygon(
        [
            [0, wave_min - range_tolerance - linear_tolerance],
            [0, wave_min + range_tolerance + linear_tolerance],
            [1000, wave_max + range_tolerance + linear_tolerance],
            [1000, wave_max - range_tolerance - linear_tolerance],
        ],
        closed=True,
        fill=True,
        alpha=0.25,
    )
)
plt.text(600.0, 5250.0, s="Search space", color="C0", alpha=0.75)

# legend and lables
plt.xlabel("Pixel")
plt.ylabel(r"Wavelength / $\AA$")

plt.xlim(0, 1000)
plt.ylim(3000, 9000)

plt.tight_layout()

plt.savefig("threshold_plot.png")
