

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np

# Directories of the models and the sample model
# Change these to your directories
# root_dir = './data/tanhgru'
root_dir = './data/debug/mnist'
model_dir = root_dir + '/1'
save_dir = './figure/mnist/1/'


fig = plt.figure(figsize=(6, 6))
for u in range(20):
    ax = fig.add_subplot(4, 5, u + 1)
    ax.plot(np.arange(10))
plt.tight_layout()
plt.savefig(save_dir +'_try.pdf')
plt.show()

fig = plt.figure(figsize=(1, 2))
heights = np.array([0.2, 0.2, 0.2])
for i in range(3):
    ax = fig.add_axes([0.2, sum(heights[i+1:]+0.08)+0.15, 0.5, heights[i]])
    plt.plot(np.arange(0,10))
    plt.yticks(np.arange(0, 10), [0, '', '', '', '', 5, '', '', '', 9], rotation='horizontal')
    plt.xticks([0, 10])
    if i == 0:
        ax.spines["right"].set_linewidth(0.5)
        ax.spines["left"].set_visible(False)

plt.tight_layout()
plt.savefig(save_dir +'_try.pdf')
plt.show()


plt.close('all')
plt.figure()
colors = pl.cm.RdPu(np.linspace(0, 2,20))
for i in range(10):
    plt.plot(np.ones(10)*i, c=colors[i], label=i)
plt.legend()
plt.savefig(save_dir +'_try.pdf')
plt.show()
