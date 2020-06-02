import matplotlib.pyplot as plt
import numpy as np

iter1 = np.arange(0.5, 5.01, 0.25)
iter2 = np.arange(0.50, 5.01, 0.50)


data_evi = np.load('lr_evi.npz')
data_svgd = np.load('lr_svgd.npz')
data_rsvgd = np.load('lr_rsvgd.npz')

evi_mean = data_evi['results_mean']
evi_var = data_evi['results_var']

svgd_mean = data_svgd['results_mean'][1:, :]
svgd_var = data_svgd['results_var'][1:, :]

rsvgd_mean = data_rsvgd['results_mean'][1:, :]
rsvgd_var = data_rsvgd['results_var'][1:, :]


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5))
ax1.errorbar(iter1, rsvgd_mean[:, 0], yerr=rsvgd_var[:, 0], fmt='o', color='orange', linestyle='--', markersize=2, label='RSVGD')
ax1.errorbar(iter2, evi_mean[:, 0], yerr=evi_var[:, 0], fmt='o', linestyle='--', markersize=2, label='EVI_Im (ours)')
ax1.errorbar(iter1, svgd_mean[:, 0], yerr=svgd_var[:, 0], fmt='o', color='green', linestyle='--', markersize=2, label='SVGD')




ax1.set_xlabel('(a) Number of Epoches', fontsize=15)
ax1.set_ylabel('Test accuracy', fontsize=12)
ax1.tick_params(axis='x', labelsize=5)
ax1.tick_params(axis='y', labelsize=5)

ax1.axvline(x=0.50, linewidth=.3, color='gray', alpha=0.5)
ax = ax1.get_xticks
ax = np.append(ax, 0.5)
# ax1.set_xticks(ax)
# ax1.axvline(x=1.0, linewidth=.3, color='gray', alpha=0.5)
# ax1.axvline(x=2.0, linewidth=.3, color='gray', alpha=0.5)
# ax1.axhline(y=0.75, linewidth=.3, color='gray', alpha=0.5)
# ax1.set_xticks(np.array([0.1, 0.1, 1.0, 2.0]))
# ax1.set_ylim([0.6, 0.76])
# ax1.set_ylim([0.6, 0.85])
# ax1.yaxis.tick_left()
ax2.errorbar(iter1, rsvgd_mean[:, 1], yerr=rsvgd_var[:, 1], fmt='o', color='orange', linestyle='--', markersize=2, label='RSVGD')
ax2.errorbar(iter2, evi_mean[:, 1], yerr=evi_var[:, 1], fmt='o', linestyle='--', markersize=2, label='EVI_Im (ours)')
ax2.errorbar(iter1, svgd_mean[:, 1], yerr=svgd_var[:, 1], fmt='o', color='green', linestyle='--', markersize=2, label='SVGD')


ax2.set_xlabel('(b) Number of Epoches', fontsize=15)
ax2.set_ylabel('Test log_likelihood', fontsize=12)
ax2.tick_params(axis='x', labelsize=5)
ax2.tick_params(axis='y', labelsize=5)
ax2.axvline(x=0.5, linewidth=.3, color='gray', alpha=0.5)
# ax2.axvline(x=0.5, linewidth=.3, color='gray', alpha=0.5)
# ax2.axvline(x=1.0, linewidth=.3, color='gray', alpha=0.5)
# ax2.axvline(x=2.0, linewidth=.3, color='gray', alpha=0.5)
# ax2.set_ylim([-0.9, -0.4])
# ax2.set_xticks(np.array([0.1, 0.1, 1.0, 2.0]))
ax2.tick_params(axis='x', labelsize=5)
ax2.tick_params(axis='y', labelsize=5)

# ax1.axhline(y=0.75, linewidth=0.8, color='black', alpha=0.6)
#ax2.axhline(y=0.75, linewidth=0.8, color='black', alpha=0.6)

# ax2.yaxis.tick_right()
#ax2.yaxis.set_label_position("right")
ax1.legend()
ax2.legend()

ax1.legend()
xt = ax1.get_xticks()
xt = np.append(xt, 50)
# ax1.set_xticks(xt)
# ax2.set_xticks(xt)


plt.show()
f.savefig('lr.png')
