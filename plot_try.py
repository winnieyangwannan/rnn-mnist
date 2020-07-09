
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np

# Directories of the models and the sample model
# Change these to your directories
# root_dir = './data/tanhgru'
root_dir = './data/debug/mnist'
model_dir = root_dir + '/1'
save_dir = './figure/mnist/1/'

plt.close('all')
plt.figure()
colors = pl.cm.RdPu(np.linspace(0, 2,20))
for i in range(10):
    plt.plot(np.ones(10)*i, c=colors[i], label=i)
plt.legend()
plt.savefig(save_dir +'_try.pdf')
plt.show()
