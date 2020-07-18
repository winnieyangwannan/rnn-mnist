import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
from analysis.standard_analysis import *
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Directories of the models and the sample model
# Change these to your directories
# root_dir = './data/tanhgru'
root_dir = './data/mnist/'
model_dir = root_dir + '10rnn_20dt_1stim_1res_end/'
save_dir = './figure/mnist/10rnn_20dt_1stim_1res_end/'

def plot_activity_pca(model_dir, save_dir):
    model = Model(model_dir)
    hp = model.hp

    # Generate a batch of trial from the test mode
    hp['current_step'] = 0
    hp['batch_size_test'] = 100

    _, x_test, _, y_test = load_mnist_data()
    x_test = x_test[0:100]
    y_test = y_test[0:100]

    with tf.Session() as sess:
        model.restore()

        trial = generate_trials('mnist', hp, mode='test')
        feed_dict = tools.gen_feed_dict(model, trial, hp)
        h, y, y_hat = sess.run([model.h, model.y, model.y_hat], feed_dict=feed_dict)

        colors = pl.cm.tab10(np.arange(0, 10))

        fig = plt.figure(figsize=(6, 6))
        ax = Axes3D(fig)
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 9]:

            _, plot_indexs = filter_digit(x_test, y_test, i)

            u, s, vh = np.linalg.svd(h[20:, plot_indexs[0], :], full_matrices=True)
            proj_h = vh[0:3, :].dot(h[20:, plot_indexs[0], :].T)

            ax.plot3D(proj_h[0, :], proj_h[1, :], proj_h[2, :], label=i, color=colors[i])
            ax.scatter(proj_h[0, 0], proj_h[1, 0], proj_h[2, 0], marker='.', color=colors[i])
            ax.scatter(proj_h[0, -1], proj_h[1, -1], proj_h[2, -1], marker='v', color=colors[i])

        plt.legend()
        fig.savefig(save_dir + 'activity_PCA_' + '.pdf')
        plt.show()



        for i in [0, 1, 2, 3, 4, 5, 6, 7,8, 9]:
            fig = plt.figure(figsize=(6, 6))
            ax = Axes3D(fig)

            _, plot_indexs = filter_digit(x_test, y_test, i)

            for ii in plot_indexs:

                u, s, vh = np.linalg.svd(h[20:, ii, :], full_matrices=True)
                proj_h = vh[0:3, :].dot(h[20:, ii, :].T)

                ax.plot3D(proj_h[0, :], proj_h[1, :], proj_h[2, :], label=i, color=colors[i])
                ax.scatter(proj_h[0, 0], proj_h[1, 0], proj_h[2, 0], marker='.', color=colors[i])
                ax.scatter(proj_h[0, -1], proj_h[1, -1], proj_h[2, -1], marker='v', color=colors[i])

            plt.legend()
            fig.savefig(save_dir + 'activity_PCA_' + str(i) + '.pdf')
            plt.show()




plot_activity_pca(model_dir, save_dir)


fig = plt.figure(figsize=(6, 6))
for u in range(20):
    ax = fig.add_subplot(4, 5, u + 1)
    ax.plot(np.arange(10))
plt.tight_layout()
plt.savefig(save_dir + '_try.pdf')
plt.show()

fig = plt.figure(figsize=(1, 2))
heights = np.array([0.2, 0.2, 0.2])
for i in range(3):
    ax = fig.add_axes([0.2, sum(heights[i + 1:] + 0.08) + 0.15, 0.5, heights[i]])
    plt.plot(np.arange(0, 10))
    plt.yticks(np.arange(0, 10), [0, '', '', '', '', 5, '', '', '', 9], rotation='horizontal')
    plt.xticks([0, 10])
    if i == 0:
        ax.spines["right"].set_linewidth(0.5)
        ax.spines["left"].set_visible(False)

plt.tight_layout()
plt.savefig(save_dir + '_try.pdf')
plt.show()

plt.close('all')
plt.figure()
colors = pl.cm.RdPu(np.linspace(0, 2, 20))
for i in range(10):
    plt.plot(np.ones(10) * i, c=colors[i], label=i)
plt.legend()
plt.savefig(save_dir + '_try.pdf')
plt.show()
