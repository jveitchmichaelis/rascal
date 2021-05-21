from matplotlib import pyplot as plt
import itertools
import matplotlib as mpl
import numpy as np
import os

# Load the LT SPRAT data
if '__file__' in locals():
    base_dir = os.path.dirname(__file__)
else:
    base_dir = os.getcwd()

colors = list(plt.cm.tab10(np.arange(10))) + ["black", "firebrick"]
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors) 

def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

max_tries = [25, 50, 75, 100, 150, 200, 250, 500, 1000, 2000, 5000]

# Number of repetance
N = 1000

best_p_mt = np.load(os.path.join(base_dir, 'best_p_mt.npy'))
rms_mt = np.load(os.path.join(base_dir, 'rms_mt.npy'))
residual_mt = np.load(os.path.join(base_dir, 'residual_mt.npy'), allow_pickle=True)
peak_utilisation_mt = np.load(os.path.join(base_dir, 'peak_utilisation_mt.npy'))
'''
best_p_manual_mt = np.load(os.path.join(base_dir, 'best_p_manual_mt.npy'))
rms_manual_mt = np.load(os.path.join(base_dir, 'rms_manual_mt.npy'))
residual_manual_mt = np.load(os.path.join(base_dir, 'residual_manual_mt.npy'), allow_pickle=True)
peak_utilisation_manual_mt = np.load(
    os.path.join(base_dir, 'peak_utilisation_manual_mt.npy'))
'''

best_p_manual_mt = np.load(os.path.join(base_dir, 'best_p_mt.npy'))
rms_manual_mt = np.load(os.path.join(base_dir, 'rms_mt.npy'))
residual_manual_mt = np.load(os.path.join(base_dir, 'residual_mt.npy'), allow_pickle=True)
peak_utilisation_manual_mt = np.load(
    os.path.join(base_dir, 'peak_utilisation_mt.npy'))

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

fig2, ax2 = plt.subplots(2, 5, sharey=True)
fig2.set_figheight(10)
fig2.set_figwidth(10)
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


ax2[0, 0].grid()
ax2[0, 1].grid()
ax2[0, 2].grid()
ax2[0, 3].grid()
ax2[0, 4].grid()

ax2[1, 0].grid()
ax2[1, 1].grid()
ax2[1, 2].grid()
ax2[1, 3].grid()
ax2[1, 4].grid()

ax2[0, 0].set_title('p0')
ax2[0, 1].set_title('p1')
ax2[0, 2].set_title('p2')
ax2[0, 3].set_title('p3')
ax2[0, 4].set_title('p4')

ax2[0, 0].set_ylabel('Auto lines')
ax2[1, 0].set_ylabel('Manual lines')

handles, labels = ax2[0, 0].get_legend_handles_labels()

fig2.legend(flip(handles, 6), flip(labels, 6), loc='lower center', mode='expand', ncol=6)
fig2.tight_layout(w_pad=0.01)
fig2.subplots_adjust(bottom=0.1)

fig2.savefig('figure_2_polynomial_coefficients.png')

# Figure 3 - RMS
fig3, ax3 = plt.subplots(2, 1, sharex=True, sharey=True)
fig3.set_figheight(6)
fig3.set_figwidth(6)

ax3[0].violinplot(rms_mt.T, showmedians=True)
ax3[1].violinplot(rms_manual_mt.T, showmedians=True)

ax3[0].set_xticks(range(1, len(max_tries)+1))
ax3[0].set_xticklabels(max_tries)
ax3[1].set_xticks(range(1, len(max_tries)+1))
ax3[1].set_xticklabels(max_tries)

ax3[0].grid()
ax3[1].grid()

ax3[0].set_ylabel(r'RMS / $\AA$')
ax3[1].set_xlabel('max_tries')
ax3[1].set_ylabel(r'RMS / $\AA$')

fig3.tight_layout()
fig3.savefig('figure_3_rms.png')

# Figure 4 - Peak Utilisation
fig4, ax4 = plt.subplots(2, 1, sharex=True, sharey=True)
fig4.set_figheight(6)
fig4.set_figwidth(6)

ax4[0].violinplot(peak_utilisation_mt.T * 100.,
                        showmedians=True)
ax4[1].violinplot(peak_utilisation_manual_mt.T * 100., showmedians=True)

ax4[0].set_xticks(range(1, len(max_tries)+1))
ax4[0].set_xticklabels(max_tries)
ax4[1].set_xticks(range(1, len(max_tries)+1))
ax4[1].set_xticklabels(max_tries)

ax4[0].grid()
ax4[1].grid()

ax4[1].set_xlabel('peak_utilisation')
ax4[0].set_ylabel('Percentage')
ax4[1].set_ylabel('Percentage')

fig4.tight_layout()
fig4.savefig('figure_4_peak_utilisation.png')

# Figure 5 - wavelength at chosen pixels
# Note: start counting from ZERO
pix = [0., 255., 511., 767., 1023.]
wave = np.array([np.array([np.zeros(len(pix))] * N)] * len(max_tries))
wave_manual = np.array([np.array([np.zeros(len(pix))] * N)] * len(max_tries))
for i, mt in enumerate(max_tries):
    for j in range(N):
        best_p = best_p_mt[i][j]
        best_p_manual = best_p_manual_mt[i][j]
        wave[i][j] = np.polynomial.polynomial.polyval(pix, best_p)
        wave_manual[i][j] = np.polynomial.polynomial.polyval(
            pix, best_p_manual)


pix0_range = np.nanpercentile(wave[-1][:, 0], [1., 99.])
pix1_range = np.nanpercentile(wave[-1][:, 1], [1., 99.])
pix2_range = np.nanpercentile(wave[-1][:, 2], [1., 99.])
pix3_range = np.nanpercentile(wave[-1][:, 3], [1., 99.])
pix4_range = np.nanpercentile(wave[-1][:, 4], [1., 99.])
pix0_range_manual = np.nanpercentile(wave_manual[-1][:, 0], [1., 99.])
pix1_range_manual = np.nanpercentile(wave_manual[-1][:, 1], [1., 99.])
pix2_range_manual = np.nanpercentile(wave_manual[-1][:, 2], [1., 99.])
pix3_range_manual = np.nanpercentile(wave_manual[-1][:, 3], [1., 99.])
pix4_range_manual = np.nanpercentile(wave_manual[-1][:, 4], [1., 99.])


fig5, ax5 = plt.subplots(2, 5, sharey=True)
fig5.set_figheight(10)
fig5.set_figwidth(10)
for i, mt in enumerate(max_tries):
    # First row - auto lines
    ax5[0, 0].hist(wave[i][:, 0],
                   bins=50,
                   range=pix0_range,
                   histtype='step',
                   label=str(mt))
    ax5[0, 1].hist(wave[i][:, 1], bins=50, range=pix1_range, histtype='step')
    ax5[0, 2].hist(wave[i][:, 2], bins=50, range=pix2_range, histtype='step')
    ax5[0, 3].hist(wave[i][:, 3], bins=50, range=pix3_range, histtype='step')
    ax5[0, 4].hist(wave[i][:, 4], bins=50, range=pix4_range, histtype='step')
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

ax5[1, 2].set_xlabel('Wavelength / A')
ax5[0, 0].set_ylabel('Frequency')
ax5[1, 0].set_ylabel('Frequency')

handles, labels = ax5[0, 0].get_legend_handles_labels()

fig5.legend(flip(handles, 6), flip(labels, 6), loc='lower center', mode='expand', ncol=6)
fig5.tight_layout(w_pad=0.01)
fig5.subplots_adjust(bottom=0.1)
fig5.savefig('figure_5_wavelengths.png')




# Figure 6 - 2D heatmap of the solution
fig6, ax6 = plt.subplots(2, 1, sharex=True, sharey=True)
fig6.set_figheight(6)
fig6.set_figwidth(6)

pix = np.arange(1024) + 1
wave = np.zeros((N, len(pix)))
wave_manual = np.zeros((N, len(pix)))

i = 10
for j in range(N):
    best_p = best_p_mt[i][j]
    best_p_manual = best_p_manual_mt[i][j]
    wave[j] = np.polynomial.polynomial.polyval(pix, best_p)
    wave_manual[j] = np.polynomial.polynomial.polyval(pix, best_p_manual)

delta_wave = wave - np.nanmedian(wave, axis=0)
delta_wave_manual = wave_manual - np.nanmedian(wave_manual, axis=0)

dw_min, dw_max = -20., 20.
dw_manual_min, dw_manual_max = -20., 20.

delta_wave_heatmap = []
delta_wave_manual_heatmap = []

for i in range(N):
    delta_wave_heatmap.append(
        np.histogram(delta_wave[:, i], bins=40, range=(dw_min, dw_max))[0])
    delta_wave_manual_heatmap.append(
        np.histogram(delta_wave_manual[:, i],
                     bins=40,
                     range=(dw_manual_min, dw_manual_max))[0])

dw_yedges = np.histogram(delta_wave, bins=40, range=(dw_min, dw_max))[1]
dw_manual_yedges = np.histogram(delta_wave_manual,
                                bins=40,
                                range=(dw_manual_min, dw_manual_max))[1]

ax6[0].imshow(np.array(delta_wave_heatmap).T, origin='lower', aspect='auto')
ax6[0].set_yticks(np.linspace(0, 40, len(dw_yedges[::10]))-0.5)
ax6[0].set_yticklabels(dw_yedges[::10].astype('int'))
ax6[0].set_ylabel(r'$\Delta\lambda\ /\ \AA$')

ax6a = ax6[0].twiny()
ax6a.set_xticks(pix[::200]-1)
ax6a.set_xticklabels(np.nanmedian(wave, axis=0)[::200].astype('int'))
ax6a.set_xlabel(r'Wavelength / $\AA$')

ax6[1].imshow(np.array(delta_wave_manual_heatmap).T, origin='lower', aspect='auto')
ax6[1].set_yticks(np.linspace(0, 40, len(dw_manual_yedges[::10]))-0.5)
ax6[1].set_yticklabels(dw_manual_yedges[::10].astype('int'))

ax6[1].set_ylabel(r'$\Delta\lambda\ /\ \AA$')
ax6[1].set_xticks(pix[::200]-1)
ax6[1].set_xticklabels((pix[::200]-1).astype('int'))
ax6[1].set_xlabel('Pixel')

fig6.tight_layout()
fig6.subplots_adjust(hspace=0.05)
fig6.savefig('figure_6_heatmap.png')

