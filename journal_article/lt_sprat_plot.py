from matplotlib import pyplot as plt
import numpy as np
import os

# Load the LT SPRAT data
if '__file__' in locals():
    base_dir = os.path.dirname(__file__)
else:
    base_dir = os.getcwd()

max_tries = [25, 50, 75, 100, 150, 200, 250, 500, 1000, 2000, 5000]

# Number of repetance
N = 1000

best_p_mt = np.load(os.path.join(base_dir, 'best_p_mt'))
rms_mt = np.load(os.path.join(base_dir, 'rms_mt'))
residual_mt = np.load(os.path.join(base_dir, 'residual_mt'))
peak_utilisation_mt = np.load(os.path.join(base_dir, 'peak_utilisation_mt'))
'''
best_p_manual_mt = np.load(os.path.join(base_dir, 'best_p_manual_mt'))
rms_manual_mt = np.load(os.path.join(base_dir, 'rms_manual_mt'))
residual_manual_mt = np.load(os.path.join(base_dir, 'residual_manual_mt'))
peak_utilisation_manual_mt = np.load(
    os.path.join(base_dir, 'peak_utilisation_manual_mt'))
'''

best_p_manual_mt = np.load(os.path.join(base_dir, 'best_p_mt'))
rms_manual_mt = np.load(os.path.join(base_dir, 'rms_mt'))
residual_manual_mt = np.load(os.path.join(base_dir, 'residual_mt'))
peak_utilisation_manual_mt = np.load(
    os.path.join(base_dir, 'peak_utilisation_mt'))

# Figure 2 - polynomial coefficients
p0_range = np.nanpercentile(best_p_mt[-1][:, 0], [1., 99.])
p1_range = np.nanpercentile(best_p_mt[-1][:, 1], [1., 99.])
p2_range = np.nanpercentile(best_p_mt[-1][:, 2], [1., 99.])
p3_range = np.nanpercentile(best_p_mt[-1][:, 3], [1., 99.])
p4_range = np.nanpercentile(best_p_mt[-1][:, 4], [1., 99.])
p0_range_manual = np.nanpercentile(best_p_manual_mt[-1][:, 0], [1., 99.])
p1_range_manual = np.nanpercentile(best_p_manual_mt[-1][:, 1], [1., 99.])
p2_range_manual = np.nanpercentile(best_p_manual_mt[-1][:, 2], [1., 99.])
p3_range_manual = np.nanpercentile(best_p_manual_mt[-1][:, 3], [1., 99.])
p4_range_manual = np.nanpercentile(best_p_manual_mt[-1][:, 4], [1., 99.])

fig2, ax2 = plt.subplot(2, 5, sharey=True)
for i, mt in enumerate(max_tries):
    # First row - auto lines
    ax2[0, 0].hist(best_p_mt[i][:, 0],
                   bins=50,
                   range=p0_range,
                   histtype='step',
                   label=str(mt))
    ax2[0, 1].hist(best_p_mt[i][:, 1],
                   bins=50,
                   range=p1_range,
                   histtype='step')
    ax2[0, 2].hist(best_p_mt[i][:, 2],
                   bins=50,
                   range=p2_range,
                   histtype='step')
    ax2[0, 3].hist(best_p_mt[i][:, 3],
                   bins=50,
                   range=p3_range,
                   histtype='step')
    ax2[0, 4].hist(best_p_mt[i][:, 4],
                   bins=50,
                   range=p4_range,
                   histtype='step')
    # Second row - manual lines
    ax2[1, 0].hist(best_p_manual_mt[i][:, 0],
                   bins=50,
                   range=p0_range_manual,
                   histtype='step')
    ax2[1, 1].hist(best_p_manual_mt[i][:, 1],
                   bins=50,
                   range=p1_range_manual,
                   histtype='step')
    ax2[1, 2].hist(best_p_manual_mt[i][:, 2],
                   bins=50,
                   range=p2_range_manual,
                   histtype='step')
    ax2[1, 3].hist(best_p_manual_mt[i][:, 3],
                   bins=50,
                   range=p3_range_manual,
                   histtype='step')
    ax2[1, 4].hist(best_p_manual_mt[i][:, 4],
                   bins=50,
                   range=p4_range_manual,
                   histtype='step')

ax2[0, 0].set_title('p0')
ax2[0, 1].set_title('p1')
ax2[0, 2].set_title('p2')
ax2[0, 3].set_title('p3')
ax2[0, 4].set_title('p4')

ax2[0, 0].set_ylabel('Auto lines')
ax2[1, 0].set_ylabel('Manual lines')

fig2.set_xlabel('Value of the polynomial coefficients')
fig2.legend(loc='bottom')
fig2.tight_layout()
fig2.savefig('figure_2_polynomial_coefficients.png')

# Figure 3 - RMS
fig3, ax3 = plt.subplot(1, 2, sharex=True, sharey=True)
for i, mt in enumerate(max_tries):
    ax3[0, 0].violinplot(rms_mt[i], showmedians=True, label=str(mt))
    ax3[0, 1].violinplot(rms_manual_mt[i], showmedians=True)

ax3[0, 0].grid()
ax3[0, 1].grid()

fig3.set_xlabel(r'\texttt{max_tries}')
fig3.set_ylabel('Frequency')
fig3.legend(loc='bottom')
fig3.tight_layout()
fig3.savefig('figure_3_rms.png')

# Figure 4 - Peak Utilisation
fig4, ax4 = plt.subplot(1, 2, sharex=True, sharey=True)
for i, mt in enumerate(max_tries):
    ax4[0, 0].violinplot(peak_utilisation_mt[i],
                         showmedians=True,
                         label=str(mt))
    ax4[0, 1].violinplot(peak_utilisation_manual_mt[i], showmedians=True)

ax4[0, 0].grid()
ax4[0, 1].grid()

fig4.set_xlabel(r'\texttt{peak_utilisation}')
fig4.set_ylabel('Percentage')
fig4.legend(loc='bottom')
fig4.tight_layout()
fig4.savefig('figure_4_peak_utilisation.png')

# Figure 5 - wavelength at chosen pixels
# Note: start counting from ZERO
pix = [0., 255., 511., 767., 1023.]
wave = [[np.zeros(N)] * len(pix)]
wave_manual = [[np.zeros(N)] * len(pix)]

pix0_range = np.nanpercentile(best_p_mt[-1][:, 0], [1., 99.])
pix1_range = np.nanpercentile(best_p_mt[-1][:, 1], [1., 99.])
pix2_range = np.nanpercentile(best_p_mt[-1][:, 2], [1., 99.])
pix3_range = np.nanpercentile(best_p_mt[-1][:, 3], [1., 99.])
pix4_range = np.nanpercentile(best_p_mt[-1][:, 4], [1., 99.])
pix0_range_manual = np.nanpercentile(best_p_manual_mt[-1][:, 0], [1., 99.])
pix1_range_manual = np.nanpercentile(best_p_manual_mt[-1][:, 1], [1., 99.])
pix2_range_manual = np.nanpercentile(best_p_manual_mt[-1][:, 2], [1., 99.])
pix3_range_manual = np.nanpercentile(best_p_manual_mt[-1][:, 3], [1., 99.])
pix4_range_manual = np.nanpercentile(best_p_manual_mt[-1][:, 4], [1., 99.])

for i, mt in enumerate(max_tries):
    for j in range(N):
        best_p = best_p_mt[i][j]
        best_p_manual = best_p_manual_mt[i][j]
        for k, w in enumerate(wave):
            wave[k][j] = np.polynomial.polynomial.polyval(pix[k], best_p)
            wave_manual[k][j] = np.polynomial.polynomial.polyval(
                pix[k], best_p_manual)

fig5, ax5 = plt.subplot(2, 5, sharey=True)
for i, mt in enumerate(max_tries):

    # First row - auto lines
    ax5[0, 0].hist(wave[i][:, 0],
                   bins=50,
                   range=pix0_range,
                   histtype='step',
                   label=str(mt))
    ax5[0, 1].hist(wave[i][:, 0], bins=50, range=pix1_range, histtype='step')
    ax5[0, 2].hist(wave[i][:, 0], bins=50, range=pix2_range, histtype='step')
    ax5[0, 3].hist(wave[i][:, 0], bins=50, range=pix3_range, histtype='step')
    ax5[0, 4].hist(wave[i][:, 0], bins=50, range=pix4_range, histtype='step')

    # Second row - manual lines
    ax5[1, 0].hist(wave_manual[i][:, 0],
                   bins=50,
                   range=pix0_range_manual,
                   histtype='step')
    ax5[1, 1].hist(wave_manual[i][:, 1],
                   bins=50,
                   range=pix1_range_manual,
                   histtype='step')
    ax5[1, 2].hist(wave_manual[i][:, 2],
                   bins=50,
                   range=pix2_range_manual,
                   histtype='step')
    ax5[1, 3].hist(wave_manual[i][:, 3],
                   bins=50,
                   range=pix3_range_manual,
                   histtype='step')
    ax5[1, 4].hist(wave_manual[i][:, 4],
                   bins=50,
                   range=pix4_range_manual,
                   histtype='step')

ax5[0, 0].grid()
ax5[0, 1].grid()
ax5[0, 2].grid()
ax5[0, 3].grid()
ax5[0, 4].grid()

ax5[1, 0].grid()
ax5[1, 1].grid()
ax5[1, 2].grid()
ax5[1, 3].grid()
ax5[1, 4].grid()

fig5.set_xlabel('Wavelength / A')
fig5.set_ylabel('Frequency')
fig5.legend(loc='bottom')
fig5.tight_layout()
fig5.savefig('figure_5_wavelengths.png')


# Figure 6 - 2D heatmap of the solution
fig6, ax6 = plt.subplot(1, 2, sharex=True, sharey=True)

pix = np.arange(1024) + 1
wave = np.zeros((N, len(pix)))
wave_manual = np.zeros((N, len(pix)))

i = 8
for j in range(N):
    best_p = best_p_mt[i][j]
    best_p_manual = best_p_manual_mt[i][j]
    for k, p in enumerate(pix):
        wave[k][j] = np.polynomial.polynomial.polyval(p, best_p)
        wave_manual[k][j] = np.polynomial.polynomial.polyval(
            p, best_p_manual)

delta_wave = wave - np.nanmedian(wave, axis=1)
delta_wave_manual = wave_manual - np.nanmedian(wave, axis=1)

dw_min, dw_max = np.nanpercentile(delta_wave, [1., 99.])
dw_manual_min, dw_manual_max = np.nanpercentile(delta_wave_manual, [1., 99.])

delta_wave_heatmap = []
delta_wave_manual_heatmap = []

for i in range(N):
    delta_wave_heatmap.append(np.histogram(delta_wave[:, 0], bins=50, range=(dw_min, dw_max))[0])
    delta_wave_manual_heatmap.append(np.histogram(delta_wave_manual[:, 0], bins=50, range=(dw_manual_min, dw_manual_max))[0])


dw_yedges = np.histogram(delta_wave[:, 0], bins=50, range=(dw_min, dw_max))[0]
dw_manual_yedges = np.histogram(delta_wave_manual[:, 0], bins=50, range=(dw_manual_min, dw_manual_max))[0]

ax6[0, 0].imshow(delta_wave_heatmap, origin='lower', aspect='auto')
ax6[0, 1].imshow(delta_wave_manual_heatmap, origin='lower', aspect='auto')

ax6[0, 0].set_yticklabels(dw_yedges)
ax6[0, 1].set_yticklabels(dw_manual_yedges)

ax6[0, 0].set_ylabels(r'$\Delta(\lambda) / \AA$')
ax6[0, 1].set_ylabels(r'$\Delta(\lambda) / \AA$')

ax6[0, 1].set_xlabels('Pixel')
ax6b = ax6[0, 1].twiny()
ax6b.set_xticks(wave[::200])
ax6b.set_xticklabels(wavewave[::200])
ax6b.set_xlabels(r'Wavelength / $\AA$')

fig6.tight_layout()
