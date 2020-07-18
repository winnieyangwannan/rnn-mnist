"""Standard analyses that can be performed on any task"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from task import generate_trials, rule_name
from network import Model
import tools
from matplotlib.pyplot import *
from mnist_task import *
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D

def pretty_singleneuron_plot_mnist(model_dir,save_dir, plot_type):
    """Plot the activity of a single neuron in time across many trials

    Args:
        model_dir:
        rules: rules to plot
        neurons: indices of neurons to plot
        epoch: epoch to plot
        save: save figure?
        ylabel_firstonly: if True, only plot ylabel for the first rule in rules
    """

    model = Model(model_dir)
    hp = model.hp
    # TODO: change later
    hp['off'] = -1
    colors = pl.cm.tab10(np.arange(0, 10))

    with tf.Session() as sess:
        model.restore()

        t_start = int(500/hp['dt'])

        # Generate a batch of trial from the test mode
        hp['current_step'] = 0
        plot_batch_size = 1000
        hp['batch_size_test'] = plot_batch_size

        _, x_test, _, y_test = load_mnist_data()
        x_test = x_test[0:plot_batch_size]
        y_test = y_test[0:plot_batch_size]

        trial = generate_trials('mnist', hp, mode='test')
        feed_dict = tools.gen_feed_dict(model, trial, hp)
        h, y, y_hat = sess.run([model.h, model.y, model.y_hat], feed_dict=feed_dict)

        #indexes_test = index_each_category(y[-1][:,1:],max_num=1.05)
        #indexes_test['0']

    stim_on = int(400 / hp['dt'])
    stim_off = int((400 + 600 * hp['stim_times']) / hp['dt'])

    fig = plt.figure(figsize=(6, 6))
    for u in range(hp['n_rnn']):
        ax = fig.add_subplot(4, 5, u + 1)
        #t_plot = np.arange(h[:].shape[0]) * hp['dt'] / 1000  # CHANGE LATER

        if plot_type == 'plot_all':
            for i in range(10):
                _, plot_indexs = filter_digit(x_test, y_test, i)
                #plot_indexs = indexes_test[str(i)][0]
                for ii in plot_indexs:
                    ax.plot(h[:, ii, u], lw=0.5, c=colors[i])

        # Plot stimulus averaged trace
        if plot_type == 'plot_average':
            # plot shaded area between stim on ad stim off
           ax.axvspan(stim_on, stim_off, color='Lavender', alpha=0.5, edgecolor='None')

           for i in range(10):
               _, plot_indexs = filter_digit(x_test, y_test, i)
               ax.plot(h[:, plot_indexs,  u].mean(axis=1), lw=1, c=colors[i], label=str(i))


        fs = 6
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')


        ax.set_xlabel('Time (s)', fontsize=fs)
        ax.set_ylabel('')

        ax.set_xticks([0, len(h)])
        ax.set_xticklabels([0, len(h) * hp['dt'] / 1000])
        ax.set_yticks([0, np.round(np.max(h[t_start:, :, u]), 2)])
        ax.set_title('')
        plt.legend(ncol=2)

        if plot_type == 'plot_average':
            fig.savefig(save_dir + 'hidden_single_unit_average.pdf')
        else:
            fig.savefig(save_dir + 'hidden_single_unit_all_trials.pdf')


        plt.tight_layout()
        fig.show()

    # plot output layer unit
    fig = plt.figure(figsize=(6, 3))
    for u in range(10):
        ax = fig.add_subplot(2, 5, u + 1)
        fs = 6
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        ax.set_xlabel('Time (s)', fontsize=fs)
        ax.set_ylabel('')

        ax.set_xticks([0, len(h)])
        ax.set_xticklabels([0, len(h) * hp['dt'] / 1000])
        ax.set_yticks(np.arange(0, 1.5, 0.5))
        ax.set_title('')
        # t_plot = np.arange(h[:].shape[0]) * hp['dt'] / 1000  # CHANGE LATER

        if plot_type == 'plot_all':
            for i in range(10):
                _, plot_indexs = filter_digit(x_test, y_test, i)
                # plot_indexs = indexes_test[str(i)][0]
                for ii in plot_indexs:
                    ax.plot(y_hat[:, ii, u], lw=0.5, c=colors[i])

        # Plot stimulus averaged trace
        if plot_type == 'plot_average':
            # plot shaded area between stim on ad stim off
            ax.axvspan(stim_on, stim_off, color='Lavender', alpha=0.5, edgecolor='None')

            for i in range(10):
                _, plot_indexs = filter_digit(x_test, y_test, i)
                ax.plot(y_hat[:, plot_indexs, u].mean(axis=1), lw=1, c=colors[i], label=str(i))


    plt.legend(ncol=2)
    plt.tight_layout()

    if plot_type == 'plot_average':
        fig.savefig(save_dir + 'out_single_unit_average.pdf')
    else:
        fig.savefig(save_dir + 'out_single_unit_all_trials.pdf')

    fig.show()


# TODO: winnie added
def schematic_plot_mnist(model_dir, save_dir, plot_time, rule=None):
    fontsize = 6
    cmap = 'Purples'

    model = Model(model_dir, dt=1)
    hp = model.hp
    hp['batch_size_test'] = 100
    hp['current_step'] = 0

    with tf.Session() as sess:
        model.restore()

        trial = generate_trials(rule, hp, mode='test')
        feed_dict = tools.gen_feed_dict(model, trial, hp)
        x = trial.x
        # TODO: Winnie added
        y = trial.y
        h, y_hat = sess.run([model.h, model.y_hat], feed_dict=feed_dict)

        # Plot input and output
        # TODO: WIinnie added

        for ii in range(hp['batch_size_test']):
            fig = plt.figure(figsize=(1, 2))
            heights = np.array([0.2, 0.2, 0.2])
            for i in range(3):
                ax = fig.add_axes([0.2, sum(heights[i+1:]+0.08)+0.15, 0.5, heights[i]])
                plt.xticks([])

                # Fixed style for these plots
                ax.tick_params(axis='both', which='major', labelsize=fontsize,
                               width=0.5, length=2, pad=3)
                ax.spines["right"].set_linewidth(0.5)
                ax.spines["left"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('right')

                if i == 0:
                    # TODO: need to change when to plot
                     test_img = x[int(plot_time / hp['dt']), ii, :].reshape(28, -1)
                     plt.imshow(test_img, cmap=cmap)
                     plt.title('Input', fontsize=fontsize, y=0.9)
                     ax.spines["right"].set_visible(False)
                     plt.yticks([])
                     plt.xticks([])



                elif i == 1:
                    plt.imshow(y_hat[:,ii,:].T, aspect='auto', cmap=cmap,
                               vmin=0, vmax=1, interpolation='none', origin='lower')
                    plt.yticks(np.arange(0, 10), [0, '', '', '', '', 5, '', '', '', 9], rotation='horizontal')
                    plt.xticks([])
                    plt.title('Response', fontsize=fontsize, y=0.9)

                elif i == 2:
                    plt.imshow(y[:, ii, :].T, aspect='auto', cmap=cmap,
                               vmin=0, vmax=1, interpolation='none', origin='lower')
                    plt.yticks(np.arange(0, 10), [0, '', '', '', '', 5, '', '', '', 9], rotation='horizontal')
                    plt.xticks([0, len(y[:, ii, 1:])], [0, len(y[:, ii, 1:])/1000])
                    plt.xlabel('duration(s)')
                    plt.title('Target', fontsize=fontsize, y=0.9)
                    ax.spines["bottom"].set_visible(True)



            plt.savefig(save_dir + 'schematic_outputs_mnist' + str(ii) + '_.pdf', transparent=True)
            plt.show()
        plt.close('all')


def plot_connectivity(model_dir, save_dir):
    import networkx as nx

    model = Model(model_dir)
    hp = model.hp

    with tf.Session() as sess:
        model.restore()
        # get all connection weights and biases as tensorflow variables
        w_in, w_rec, w_out = sess.run([model.w_sen_in, model.w_rec, model.w_out])


    fig = plt.figure(figsize=(6, 6))
    for uu in range(hp['n_rnn']):
        ax = fig.add_subplot(4, 5, uu+1)
        plt.imshow(w_in[:, uu].reshape(28, 28))

        ax.set_title('unit ' + str(uu+1), fontsize=6)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.suptitle('W_in', fontsize=6)
    fig.savefig(save_dir + 'w_in.pdf')
    plt.show()


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


        # plot each digit in separate graph
        colors = pl.cm.tab20(np.arange(0, 20))
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            fig = plt.figure(figsize=(6, 6))
            ax = Axes3D(fig)

            _, plot_indexs = filter_digit(x_test, y_test, i)

            for idx, ii in enumerate(plot_indexs):

                u, s, vh = np.linalg.svd(h[20:, ii, :], full_matrices=True)
                proj_h = vh[0:3, :].dot(h[20:, ii, :].T)

                ax.plot3D(proj_h[0, :], proj_h[1, :], proj_h[2, :], color=colors[idx], label=ii)
                ax.scatter(proj_h[0, 0], proj_h[1, 0], proj_h[2, 0], color=colors[idx], marker='.')
                ax.scatter(proj_h[0, -1], proj_h[1, -1], proj_h[2, -1], color=colors[idx], marker='v')

            plt.legend()
            fig.savefig(save_dir + 'activity_PCA_' + str(i) + '.pdf')
            plt.show()

def easy_activity_plot(model_dir, rule):
    """A simple plot of neural activity from one task.

    Args:
        model_dir: directory where model file is saved
        rule: string, the rule to plot
    """

    model = Model(model_dir)
    hp = model.hp

    with tf.Session() as sess:
        model.restore()

        trial = generate_trials(rule, hp, mode='test')
        feed_dict = tools.gen_feed_dict(model, trial, hp)
        h, y_hat = sess.run([model.h, model.y_hat], feed_dict=feed_dict)
        # All matrices have shape (n_time, n_condition, n_neuron)

    # Take only the one example trial
    i_trial = 0

    for activity, title in zip([trial.x, h, y_hat],
                               ['input', 'recurrent', 'output']):
        plt.figure()
        plt.imshow(activity[:,i_trial,:].T, aspect='auto', cmap='hot',
                   interpolation='none', origin='lower')
        plt.title(title)
        plt.colorbar()
        plt.show()


def easy_connectivity_plot(model_dir):
    """A simple plot of network connectivity."""

    model = Model(model_dir)
    with tf.Session() as sess:
        model.restore()
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get name of each variable
        names  = [var.name for var in var_list]

    # Plot weights
    for param, name in zip(params, names):
        if len(param.shape) != 2:
            continue

        vmax = np.max(abs(param))*0.7
        plt.figure()
        # notice the transpose
        plt.imshow(param.T, aspect='auto', cmap='bwr', vmin=-vmax, vmax=vmax,
                   interpolation='none', origin='lower')
        plt.title(name)
        plt.colorbar()
        plt.xlabel('From')
        plt.ylabel('To')
        plt.show()


def pretty_inputoutput_plot(model_dir, rule, save=False, plot_ylabel=False):
    """Plot the input and output activity for a sample trial from one task.

    Args:
        model_dir: model directory
        rule: string, the rule
        save: bool, whether to save plots
        plot_ylabel: bool, whether to plot ylable
    """


    fs = 7

    model = Model(model_dir)
    hp = model.hp

    with tf.Session() as sess:
        model.restore()

        trial = generate_trials(rule, hp, mode='test')
        x, y = trial.x, trial.y
        feed_dict = tools.gen_feed_dict(model, trial, hp)
        h, y_hat = sess.run([model.h, model.y_hat], feed_dict=feed_dict)

        t_plot = np.arange(x.shape[0])*hp['dt']/1000

        assert hp['num_ring'] == 2

        n_eachring = hp['n_eachring']

        fig = plt.figure(figsize=(1.3,2))
        ylabels = ['fix. in', 'stim. mod1', 'stim. mod2','fix. out', 'out']
        heights = np.array([0.03,0.2,0.2,0.03,0.2])+0.01
        for i in range(5):
            ax = fig.add_axes([0.15,sum(heights[i+1:]+0.02)+0.1,0.8,heights[i]])
            cmap = 'Purples'
            plt.xticks([])
            ax.tick_params(axis='both', which='major', labelsize=fs,
                           width=0.5, length=2, pad=3)

            if plot_ylabel:
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')

            else:
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.xaxis.set_ticks_position('none')

            if i == 0:
                plt.plot(t_plot, x[:,0,0], color='xkcd:blue')
                if plot_ylabel:
                    plt.yticks([0,1],['',''],rotation='vertical')
                plt.ylim([-0.1,1.5])
                plt.title(rule_name[rule],fontsize=fs)
            elif i == 1:
                plt.imshow(x[:,0,1:1+n_eachring].T, aspect='auto', cmap=cmap,
                           vmin=0, vmax=1, interpolation='none',origin='lower')
                if plot_ylabel:
                    plt.yticks([0, (n_eachring-1)/2, n_eachring-1],
                               [r'0$\degree$',r'180$\degree$',r'360$\degree$'],
                               rotation='vertical')
            elif i == 2:
                plt.imshow(x[:, 0, 1+n_eachring:1+2*n_eachring].T,
                           aspect='auto', cmap=cmap, vmin=0, vmax=1,
                           interpolation='none',origin='lower')

                if plot_ylabel:
                    plt.yticks(
                        [0, (n_eachring-1)/2, n_eachring-1],
                        [r'0$\degree$', r'180$\degree$', r'360$\degree$'],
                        rotation='vertical')
            elif i == 3:
                plt.plot(t_plot, y[:,0,0],color='xkcd:green')
                plt.plot(t_plot, y_hat[:,0,0],color='xkcd:blue')
                if plot_ylabel:
                    plt.yticks([0.05,0.8],['',''],rotation='vertical')
                plt.ylim([-0.1,1.1])
            elif i == 4:
                plt.imshow(y_hat[:, 0, 1:].T, aspect='auto', cmap=cmap,
                           vmin=0, vmax=1, interpolation='none', origin='lower')
                if plot_ylabel:
                    plt.yticks(
                        [0, (n_eachring-1)/2, n_eachring-1],
                        [r'0$\degree$', r'180$\degree$', r'360$\degree$'],
                        rotation='vertical')
                plt.xticks([0,y_hat.shape[0]], ['0', '2'])
                plt.xlabel('Time (s)',fontsize=fs, labelpad=-3)
                ax.spines["bottom"].set_visible(True)

            if plot_ylabel:
               plt.ylabel(ylabels[i],fontsize=fs)
            else:
                plt.yticks([])
            ax.get_yaxis().set_label_coords(-0.12,0.5)

        if save:
            save_name = 'figure/sample_'+rule_name[rule].replace(' ','')+'.pdf'
            plt.savefig(save_name, transparent=True)
        plt.show()


# TODO: winnie added
def pretty_singleneuron_plot_mnist_old(model_dir,save_dir, plot_type):
    """Plot the activity of a single neuron in time across many trials

    Args:
        model_dir:
        rules: rules to plot
        neurons: indices of neurons to plot
        epoch: epoch to plot
        save: save figure?
        ylabel_firstonly: if True, only plot ylabel for the first rule in rules
    """

    model = Model(model_dir)
    hp = model.hp
    colors = pl.cm.RdPu(np.linspace(0, 1, 10))

    with tf.Session() as sess:
        model.restore()

        t_start = int(500/hp['dt'])

        # Generate a batch of trial from the test mode
        hp['current_step'] = 0
        hp['batch_size_test'] = 400

        _, x_test, _, y_test = load_mnist_data()
        x_test = x_test[0:400]
        y_test = y_test[0:400]

        trial = generate_trials('mnist', hp, mode='test')
        feed_dict = tools.gen_feed_dict(model, trial, hp)
        h, y, y_hat = sess.run([model.h, model.y, model.y_hat], feed_dict=feed_dict)



    for p in range(int(400 / 40)):
        fig = plt.figure(figsize=(6, 6))
        for u in range(40):
            ax = fig.add_subplot(8, 5, u + 1)
            t_plot = np.arange(h[t_start:].shape[0]) * hp['dt'] / 1000

            if plot_type == 'plot_all':
                for i in range(10):
                    _, plot_indexs = filter_digit(x_test, y_test, i)
                    #plot_indexs = indexes_test[str(i)][0]
                    for ii in plot_indexs:
                        _ = ax.plot(t_plot, h[t_start:, ii, u+40*p], lw=0.5, c=colors[i])

            if plot_type == 'plot_average':
               # Plot stimulus averaged trace
               for i in range(10):
                   _, plot_indexs = filter_digit(x_test, y_test, i)
                   _ = ax.plot(t_plot, h[t_start:, plot_indexs,  u+40*p].mean(axis=1), lw=1, c=colors[i])

            fs = 6
            ax.tick_params(axis='both', which='major', labelsize=fs)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')


            ax.set_xlabel('Time (s)', fontsize=fs)
            ax.set_ylabel('')

            ax.set_xticks([0,1])
            ax.set_yticks([0, np.round(np.max(h[t_start:, :, u+40*p]),2)])
            ax.set_title('')


        plt.tight_layout()
        if plot_type == 'plot_average':
            plt.savefig(save_dir + str(p) + '_try_average.pdf')
        else:
            plt.savefig(save_dir + str(p) + '_try.pdf')
        plt.show()


def pretty_singleneuron_plot(model_dir,
                             rules,
                             neurons,
                             epoch=None,
                             save=False,
                             ylabel_firstonly=True,
                             trace_only=False,
                             plot_stim_avg=False,
                             save_name=''):
    """Plot the activity of a single neuron in time across many trials

    Args:
        model_dir:
        rules: rules to plot
        neurons: indices of neurons to plot
        epoch: epoch to plot
        save: save figure?
        ylabel_firstonly: if True, only plot ylabel for the first rule in rules
    """

    if isinstance(rules, str):
        rules = [rules]

    try:
        _ = iter(neurons)
    except TypeError:
        neurons = [neurons]

    h_tests = dict()
    model = Model(model_dir)
    hp = model.hp
    with tf.Session() as sess:
        model.restore()

        t_start = int(500/hp['dt'])

        for rule in rules:
            # Generate a batch of trial from the test mode
            trial = generate_trials(rule, hp, mode='test')
            feed_dict = tools.gen_feed_dict(model, trial, hp)
            h = sess.run(model.h, feed_dict=feed_dict)
            h_tests[rule] = h

    for neuron in neurons:
        h_max = np.max([h_tests[r][t_start:,:,neuron].max() for r in rules])
        for j, rule in enumerate(rules):
            fs = 6
            fig = plt.figure(figsize=(1.0,0.8))
            ax = fig.add_axes([0.35,0.25,0.55,0.55])
            t_plot = np.arange(h_tests[rule][t_start:].shape[0])*hp['dt']/1000
            _ = ax.plot(t_plot,
                        h_tests[rule][t_start:,:,neuron], lw=0.5, color='gray')

            if plot_stim_avg:
                # Plot stimulus averaged trace
                _ = ax.plot(np.arange(h_tests[rule][t_start:].shape[0])*hp['dt']/1000,
                        h_tests[rule][t_start:,:,neuron].mean(axis=1), lw=1, color='HotPink')

            if epoch is not None:
                e0, e1 = trial.epochs[epoch]
                e0 = e0 if e0 is not None else 0
                e1 = e1 if e1 is not None else h_tests[rule].shape[0]
                ax.plot([e0, e1], [h_max*1.15]*2,
                        color='HotPink',linewidth=1.5)
                figname = 'figure/trace_'+rule_name[rule]+epoch+save_name+'.pdf'
            else:
                figname = 'figure/trace_unit'+str(neuron)+rule_name[rule]+save_name+'.pdf'

            plt.ylim(np.array([-0.1, 1.2])*h_max)
            plt.xticks([0, 1.5])
            plt.xlabel('Time (s)', fontsize=fs, labelpad=-5)
            plt.locator_params(axis='y', nbins=4)
            if j>0 and ylabel_firstonly:
                ax.set_yticklabels([])
            else:
                plt.ylabel('Activitity (a.u.)', fontsize=fs, labelpad=2)
            plt.title('Unit {:d} '.format(neuron) + rule_name[rule], fontsize=5)
            ax.tick_params(axis='both', which='major', labelsize=fs)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            if trace_only:
                ax.spines["left"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.xaxis.set_ticks_position('none')
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('')

            if save:
                plt.savefig(figname, transparent=True)
            plt.show()


def plot_connectivity(model_dir, save_dir):
    import networkx as nx

    model = Model(model_dir)
    hp = model.hp

    with tf.Session() as sess:
        model.restore()
        # get all connection weights and biases as tensorflow variables
        w_in, w_rec, w_out = sess.run([model.w_sen_in, model.w_rec, model.w_out])



    # plot input connectivity
    fig = plt.figure(figsize=(6, 6))
    for uu in range(hp['n_rnn']):
        ax = fig.add_subplot(4, 5, uu+1)
        plt.imshow(w_in[:, uu].reshape(28, 28))

        ax.set_title('unit ' + str(uu+1), fontsize=6)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.suptitle('W_in', fontsize=6)
    fig.savefig(save_dir + 'w_in.pdf')
    plt.show()



    # plot recurrent connectivity
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0.1, 0.25, 0.6, 0.6])
    plt.imshow(w_rec)


    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_xticks(np.arange(0, hp['n_rnn']))
    ax.set_yticks(np.arange(0, hp['n_rnn']))
    ax.set_xticklabels(np.arange(1, hp['n_rnn']+1), rotation=90)
    ax.set_yticklabels(np.arange(1, hp['n_rnn']+1))

    plt.tight_layout()
    plt.colorbar()
    fig.suptitle('W_rec', fontsize=12)
    fig.savefig(save_dir + 'w_rec.pdf')
    plt.show()


    # plot output connectivity
    fig = plt.figure(figsize=(6, 6))
    for uu in range(hp['n_output']):
        ax = fig.add_subplot(5, 2, uu + 1)
        plt.imshow(w_out[:, uu].reshape(-1, 5))

        ax.set_title(str(uu), fontsize=6)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.suptitle('W_out', fontsize=6)
    fig.savefig(save_dir + 'w_out.pdf')
    plt.show()



def activation_patter_plot_mnist(model_dir, save_dir):
    model = Model(model_dir)
    hp = model.hp
    colors = pl.cm.RdPu(np.linspace(0, 1, 10))

    with tf.Session() as sess:
        model.restore()

        # Generate a batch of trial from the test mode
        hp['current_step'] = 0
        hp['batch_size_test'] = 1000
        trial = generate_trials('mnist', hp, mode='test')
        feed_dict = tools.gen_feed_dict(model, trial, hp)
        x, h, y, y_hat = sess.run([model.x, model.h, model.y, model.y_hat], feed_dict=feed_dict)
        #indexes_test = filter_digit(model.x, y, 3)
        #indexes_test = index_each_category(y[-1][:, 1:], max_num=1.05)

        _, x_test, _, y_test = load_mnist_data()
        x_test = x_test[0:1000]
        y_test = y_test[0:1000]

        fig = plt.figure(figsize=(6, 6))
        for i in range(10):
            ax = fig.add_subplot(5, 2, i+1)
            _, indexs = filter_digit(x_test, y_test, i)
            im = ax.imshow(h[-1, indexs, :].mean(axis=0).reshape(20, -1))
            ax.set_title(str(i))

            ax.set_xticks([])
            ax.set_yticks([])
        plt.colorbar(im)
        plt.tight_layout()
        fig.savefig(save_dir + 'activation_pattern.pdf')
        plt.show()



def activity_histogram(model_dir,
                       rules,
                       title=None,
                       save_name=None):
    """Plot the activity histogram."""

    if isinstance(rules, str):
        rules = [rules]

    h_all = None
    model = Model(model_dir)
    hp = model.hp
    with tf.Session() as sess:
        model.restore()

        t_start = int(500/hp['dt'])

        for rule in rules:
            # Generate a batch of trial from the test mode
            trial = generate_trials(rule, hp, mode='test')
            feed_dict = tools.gen_feed_dict(model, trial, hp)
            h = sess.run(model.h, feed_dict=feed_dict)
            h = h[t_start:, :, :]
            if h_all is None:
                h_all = h
            else:
                h_all = np.concatenate((h_all, h), axis=1)

    # var = h_all.var(axis=0).mean(axis=0)
    # ind = var > 1e-2
    # h_plot = h_all[:, :, ind].flatten()
    h_plot = h_all.flatten()

    fig = plt.figure(figsize=(1.5, 1.2))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.6])
    ax.hist(h_plot, bins=20, density=True)
    ax.set_xlabel('Activity', fontsize=7)
    [ax.spines[s].set_visible(False) for s in ['left', 'top', 'right']]
    ax.set_yticks([])


def schematic_plot(model_dir, save_dir, rule=None):
    fontsize = 6

    rule = rule or 'dm1'

    model = Model(model_dir, dt=1)
    hp = model.hp


    with tf.Session() as sess:
        model.restore()
        trial = generate_trials(rule, hp, mode='test')
        feed_dict = tools.gen_feed_dict(model, trial, hp)
        x = trial.x
        h, y_hat = sess.run([model.h, model.y_hat], feed_dict=feed_dict)

    n_eachring = hp['n_eachring']
    n_hidden = hp['n_rnn']

    # Plot Stimulus
    fig = plt.figure(figsize=(1.0, 1.2))
    heights = np.array([0.06, 0.25, 0.25])
    for i in range(3):
        ax = fig.add_axes([0.2, sum(heights[i + 1:] + 0.1) + 0.05, 0.7, heights[i]])
        cmap = 'Purples'
        plt.xticks([])

        # Fixed style for these plots
        ax.tick_params(axis='both', which='major', labelsize=fontsize, width=0.5, length=2, pad=3)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        if i == 0:
            plt.plot(x[:, 0, 0], color='xkcd:blue')
            plt.yticks([0, 1], ['', ''], rotation='vertical')
            plt.ylim([-0.1, 1.5])
            plt.title('Fixation input', fontsize=fontsize, y=0.9)
        elif i == 1:
            plt.imshow(x[:, 0, 1:1 + n_eachring].T, aspect='auto', cmap=cmap,
                       vmin=0, vmax=1, interpolation='none', origin='lower')
            plt.yticks([0, (n_eachring - 1) / 2, n_eachring - 1],
                       [r'0$\degree$', '', r'360$\degree$'],
                       rotation='vertical')
            plt.title('Stimulus mod 1', fontsize=fontsize, y=0.9)
        elif i == 2:
            plt.imshow(x[:, 0, 1 + n_eachring:1 + 2 * n_eachring].T, aspect='auto',
                       cmap=cmap, vmin=0, vmax=1,
                       interpolation='none', origin='lower')
            plt.yticks([0, (n_eachring - 1) / 2, n_eachring - 1], ['', '', ''],
                       rotation='vertical')
            plt.title('Stimulus mod 2', fontsize=fontsize, y=0.9)
        ax.get_yaxis().set_label_coords(-0.12, 0.5)
    plt.savefig(save_dir + 'schematic_input.pdf', transparent=True)
    plt.show()

    # Plot Rule Inputs
    fig = plt.figure(figsize=(1.0, 0.5))
    ax = fig.add_axes([0.2, 0.3, 0.7, 0.45])
    cmap = 'Purples'
    X = x[:, 0, 1 + 2 * n_eachring:]
    plt.imshow(X.T, aspect='auto', vmin=0, vmax=1, cmap=cmap,
               interpolation='none', origin='lower')

    plt.xticks([0, X.shape[0]])
    ax.set_xlabel('Time (ms)', fontsize=fontsize, labelpad=-5)

    # Fixed style for these plots
    ax.tick_params(axis='both', which='major', labelsize=fontsize,
                   width=0.5, length=2, pad=3)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.yticks([0, X.shape[-1] - 1], ['1', str(X.shape[-1])], rotation='vertical')
    plt.title('Rule inputs', fontsize=fontsize, y=0.9)
    ax.get_yaxis().set_label_coords(-0.12, 0.5)

    plt.savefig(save_dir + 'schematic_rule.pdf', transparent=True)
    plt.show()

    # Plot Units
    fig = plt.figure(figsize=(1.0, 0.8))
    ax = fig.add_axes([0.2, 0.1, 0.7, 0.75])
    cmap = 'Purples'
    plt.xticks([])
    # Fixed style for these plots
    ax.tick_params(axis='both', which='major', labelsize=fontsize,
                   width=0.5, length=2, pad=3)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.imshow(h[:, 0, :].T, aspect='auto', cmap=cmap, vmin=0, vmax=1,
               interpolation='none', origin='lower')
    plt.yticks([0, n_hidden - 1], ['1', str(n_hidden)], rotation='vertical')
    plt.title('Recurrent units', fontsize=fontsize, y=0.95)
    ax.get_yaxis().set_label_coords(-0.12, 0.5)
    plt.savefig(save_dir + 'schematic_units.pdf', transparent=True)
    plt.show()

    # Plot Outputs
    fig = plt.figure(figsize=(1.0, 0.8))
    heights = np.array([0.1, 0.45]) + 0.01
    for i in range(2):
        ax = fig.add_axes([0.2, sum(heights[i + 1:] + 0.15) + 0.1, 0.7, heights[i]])
        cmap = 'Purples'
        plt.xticks([])

        # Fixed style for these plots
        ax.tick_params(axis='both', which='major', labelsize=fontsize,
                       width=0.5, length=2, pad=3)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        if i == 0:
            plt.plot(y_hat[:, 0, 0], color='xkcd:blue')
            plt.yticks([0.05, 0.8], ['', ''], rotation='vertical')
            plt.ylim([-0.1, 1.1])
            plt.title('Fixation output', fontsize=fontsize, y=0.9)

        elif i == 1:
            plt.imshow(y_hat[:, 0, 1:].T, aspect='auto', cmap=cmap,
                       vmin=0, vmax=1, interpolation='none', origin='lower')
            plt.yticks([0, (n_eachring - 1) / 2, n_eachring - 1],
                       [r'0$\degree$', '', r'360$\degree$'],
                       rotation='vertical')
            plt.xticks([])
            plt.title('Response', fontsize=fontsize, y=0.9)

        ax.get_yaxis().set_label_coords(-0.12, 0.5)

    plt.savefig(save_dir + 'schematic_outputs.pdf', transparent=True)
    plt.show()

def networkx_illustration(model_dir, save_dir):
    import networkx as nx

    model = Model(model_dir)
    with tf.Session() as sess:
        model.restore()
        # get all connection weights and biases as tensorflow variables
        w_rec = sess.run(model.w_rec)
        
    w_rec_flat = w_rec.flatten()
    ind_sort = np.argsort(abs(w_rec_flat - np.mean(w_rec_flat)))
    n_show = int(0.01*len(w_rec_flat))
    ind_gone = ind_sort[:-n_show]
    ind_keep = ind_sort[-n_show:]
    w_rec_flat[ind_gone] = 0
    w_rec2 = np.reshape(w_rec_flat, w_rec.shape)
    w_rec_keep = w_rec_flat[ind_keep]
    G=nx.from_numpy_array(abs(w_rec2), create_using=nx.DiGraph())

    color = w_rec_keep
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    nx.draw(G,
            linewidths=0,
            width=0.1,
            alpha=1.0,
            edge_vmin=-3,
            edge_vmax=3,
            arrows=False,
            pos=nx.circular_layout(G),
            node_color=np.array([99./255]*3),
            node_size=10,
            edge_color=color,
            edge_cmap=plt.cm.RdBu_r,
            ax=ax)
    plt.savefig(save_dir + '/illustration_networkx.pdf', transparent=True)
    plt.show()


if __name__ == "__main__":
    root_dir = './data/train_all'
    model_dir = root_dir + '/0'

    # Rules to analyze
    # rule = 'dm1'
    # rule = ['dmsgo','dmsnogo','dmcgo','dmcnogo']

    # Easy activity plot, see this function to begin your analysis
    # rule = 'contextdm1'
    # easy_activity_plot(model_dir, rule)

    # Easy connectivity plot
    # easy_connectivity_plot(model_dir)

    # Plot sample activity
    # pretty_inputoutput_plot(model_dir, rule, save=False)

    # Plot a single in time
    # pretty_singleneuron_plot(model_dir, rule, [0], epoch=None, save=False,
    #                          trace_only=True, plot_stim_avg=True)

    # Plot activity histogram
    # model_dir = '/Users/guangyuyang/MyPython/RecurrentNetworkTraining/multitask/data/varyhp/33'
    # activity_histogram(model_dir, ['contextdm1', 'contextdm2'])

    # Plot schematic
    # schematic_plot(model_dir, rule)
    
    # Plot networkx illustration
    # networkx_illustration(model_dir)

    
