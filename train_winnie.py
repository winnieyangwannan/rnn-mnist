"""Main training loop"""

from __future__ import division

import sys
import time
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import task
from task import generate_trials
from network import Model, get_perf
from analysis import variance
import tools


def get_default_hp(ruleset):
    '''Get a default hp.
    Useful for debugging.
    Returns:
        hp : a dictionary containing training hpuration
    '''
    num_ring = task.get_num_ring(ruleset)
    n_rule = task.get_num_rule(ruleset)

    # TODO: winnie changed
    if ruleset == 'mnist':
        n_eachring = 784
    else:
        n_eachring = 32


    #TODO: Winnie changed
    if ruleset == 'mnist':
        n_input, n_output = 784, 10
        batch_size_train = 60
        batch_size_test = 40
        batch_size_vali = 10


    else:
        n_input, n_output = 1 + num_ring * n_eachring + n_rule, n_eachring + 1
        batch_size_train = 64
        batch_size_test = 512



    hp = {
        # batch size for training
        'batch_size_train': batch_size_train,
        # TODO: Winnie changed
        # batch_size for testing
        'batch_size_test': batch_size_test,
        'batch_size_vali': batch_size_vali,
        # input type: normal, multi
        'in_type': 'normal',
        # Type of RNNs: LeakyRNN, LeakyGRU, EILeakyGRU, GRU, LSTM
        'rnn_type': 'LeakyRNN',
        # whether rule and stimulus inputs are represented separately
        'use_separate_input': False,
        # Type of loss functions
        'loss_type': 'lsq',
        # Optimizer
        'optimizer': 'adam',
        # Type of activation runctions, relu, softplus, tanh, elu
        'activation': 'relu',
        # Time constant (ms)
        'tau': 100,
        # discretization time step (ms)
        #'dt': 20,
        # discretization time step/time constant
        'alpha': 0.2,
        # recurrent noise
        'sigma_rec': 0.05,
        # input noise
        'sigma_x': 0.01,
        # leaky_rec weight initialization, diag, randortho, randgauss
        'w_rec_init': 'randortho',
        # a default weak regularization prevents instability
        'l1_h': 0,
        # l2 regularization on activity
        'l2_h': 0,
        # l2 regularization on weight
        'l1_weight': 0,
        # l2 regularization on weight
        'l2_weight': 0,
        # l2 regularization on deviation from initialization
        'l2_weight_init': 0,
        # proportion of weights to train, None or float between (0, 1)
        'p_weight_train': None,
        # Stopping performance
        'target_perf': 1.,
        # number of units each ring
        'n_eachring': n_eachring,
        # number of rings
        'num_ring': num_ring,
        # number of rules
        'n_rule': n_rule,
        # first input index for rule units
        'rule_start': num_ring * n_eachring,
        # number of input units
        'n_input': n_input,
        # number of output units
        'n_output': n_output,
        # number of recurrent units
        'n_rnn': 256,
        # number of input units
        'ruleset': ruleset,
        # name to save
        'save_name': 'test',
        # learning rate
        'learning_rate': 0.001,
        # intelligent synapses parameters, tuple (c, ksi)
        'c_intsyn': 0,
        'ksi_intsyn': 0
        # TODO: WINNIE ADDED
        #'EPOCHS': EPOCHS
        }

    return hp

def do_eval(sess, model, log, rule_train, epoch):
    """Do evaluation.
    Args:
        sess: tensorflow session
        model: Model class instance
        log: dictionary that stores the log
        rule_train: string or list of strings, the rules being trained
    """
    hp = model.hp

    if not hasattr(rule_train, '__iter__'):
        rule_name_print = rule_train
    else:
        rule_name_print = ' & '.join(rule_train)

    print('Trial {:7d}'.format(log['trials'][-1]) +
          '  | Time {:0.2f} s'.format(log['times'][-1]) +
          '  | Now training ' + rule_name_print)


    for rule_test in hp['rules']:
        clsq_tmp = list()
        creg_tmp = list()
        perf_tmp = list()

        # TODO: WINNIE ADD, DELETE IF NO CORRECT
        if rule_test == 'mnist':
            # TODO: winnie changed batch size to 10
            trial = generate_trials(rule_test, hp, 'vali', batch_size=10)
            feed_dict = tools.gen_feed_dict(model, trial, hp)
            c_lsq, c_reg, y_hat_test = sess.run(
                [model.cost_lsq, model.cost_reg, model.y_hat],
                feed_dict=feed_dict)
            # Cost is first summed over time,
            # and averaged across batch and units
            # We did the averaging over time through c_mask
            # TODO: winnie changed
            if rule_train == ['mnist']:
                perf_test = np.mean(get_perf(y_hat_test, trial.y, hp, rule='mnist'))
            else:
                perf_test = np.mean(get_perf(y_hat_test, trial.y_loc, hp))

            clsq_tmp.append(c_lsq)
            creg_tmp.append(c_reg)
            perf_tmp.append(perf_test)

        log['cost_' + rule_test].append(np.mean(clsq_tmp, dtype=np.float64))
        log['creg_' + rule_test].append(np.mean(creg_tmp, dtype=np.float64))
        log['perf_vali_' + rule_test + str(epoch)].append(np.mean(perf_tmp, dtype=np.float64))
        print('{:15s}'.format(rule_test) +
              '| cost {:0.6f}'.format(np.mean(clsq_tmp)) +
              '| c_reg {:0.6f}'.format(np.mean(creg_tmp)) +
              '  | perf {:0.2f}'.format(np.mean(perf_tmp)))
        sys.stdout.flush()

    # Saving the model
    model.save()
    tools.save_log(log)

    return log


def display_rich_output(model, sess, step, log, model_dir):
    """Display step by step outputs during training."""
    variance._compute_variance_bymodel(model, sess)
    rule_pair = ['contextdm1', 'contextdm2']
    save_name = '_atstep' + str(step)
    title = ('Step ' + str(step) +
             ' Perf. {:0.2f}'.format(log['perf_avg'][-1]))
    variance.plot_hist_varprop(model_dir, rule_pair,
                               figname_extra=save_name,
                               title=title)
    plt.close('all')


def train(model_dir,
          hp=None,
          max_steps=1e7,
          display_step=100,
          ruleset='mante',
          rule_trains=None,
          rule_prob_map=None,
          seed=0,
          rich_output=False,
          load_dir=None,
          trainables=None,
          ):
    """Train the network.
    Args:
        model_dir: str, training directory
        hp: dictionary of hyperparameters
        max_steps: int, maximum number of training steps
        display_step: int, display steps
        ruleset: the set of rules to train
        rule_trains: list of rules to train, if None then all rules possible
        rule_prob_map: None or dictionary of relative rule probability
        seed: int, random seed to be used
    Returns:
        model is stored at model_dir/model.ckpt
        training configuration is stored at model_dir/hp.json
    """

    tools.mkdir_p(model_dir)

    # Network parameters
    # TODO: winnie add
    print(ruleset)
    hp['display_step'] = display_step

    default_hp = get_default_hp(ruleset)
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp['seed'] = seed
    hp['rng'] = np.random.RandomState(seed)

    # Rules to train and test. Rules in a set are trained together
    if rule_trains is None:
        # By default, training all rules available to this ruleset
        hp['rule_trains'] = task.rules_dict[ruleset]
    else:
        hp['rule_trains'] = rule_trains
    hp['rules'] = hp['rule_trains']

    # Assign probabilities for rule_trains.
    if rule_prob_map is None:
        rule_prob_map = dict()

    # Turn into rule_trains format
    hp['rule_probs'] = None
    if hasattr(hp['rule_trains'], '__iter__'):
        # Set default as 1.
        rule_prob = np.array(
            [rule_prob_map.get(r, 1.) for r in hp['rule_trains']])
        hp['rule_probs'] = list(rule_prob / np.sum(rule_prob))

    # TODO: Winnie add. Check if this works, if not delete later
    if ruleset == 'mnist':
        hp['rule_trains'] = ['mnist']
        hp['rules'] = hp['rule_trains']
        max_steps = 100  #because there is 60000 training examples


    tools.save_hp(hp, model_dir)

    # Build the model
    model = Model(model_dir, hp=hp)

    # Display hp
    for key, val in hp.items():
        print('{:20s} = '.format(key) + str(val))

    # Store results
    log = defaultdict(list)
    log['model_dir'] = model_dir

    # Record time
    t_start = time.time()


    with tf.Session() as sess:
        if load_dir is not None:
            model.restore(load_dir)  # complete restore
        else:
            # Assume everything is restored
            sess.run(tf.global_variables_initializer())

        # Set trainable parameters
        if trainables is None or trainables == 'all':
            var_list = model.var_list  # train everything
        elif trainables == 'input':
            # train all nputs
            var_list = [v for v in model.var_list
                        if ('input' in v.name) and ('rnn' not in v.name)]
        elif trainables == 'rule':
            # train rule inputs only
            var_list = [v for v in model.var_list if 'rule_input' in v.name]
        else:
            raise ValueError('Unknown trainables')
        model.set_optimizer(var_list=var_list)

        # penalty on deviation from initial weight
        if hp['l2_weight_init'] > 0:
            anchor_ws = sess.run(model.weight_list)
            for w, w_val in zip(model.weight_list, anchor_ws):
                model.cost_reg += (hp['l2_weight_init'] *
                                   tf.nn.l2_loss(w - w_val))

            model.set_optimizer(var_list=var_list)

        # partial weight training
        if ('p_weight_train' in hp and
                (hp['p_weight_train'] is not None) and
                hp['p_weight_train'] < 1.0):
            for w in model.weight_list:
                w_val = sess.run(w)
                w_size = sess.run(tf.size(w))
                w_mask_tmp = np.linspace(0, 1, w_size)
                hp['rng'].shuffle(w_mask_tmp)
                ind_fix = w_mask_tmp > hp['p_weight_train']
                w_mask = np.zeros(w_size, dtype=np.float32)
                w_mask[ind_fix] = 1e-1  # will be squared in l2_loss
                w_mask = tf.constant(w_mask)
                w_mask = tf.reshape(w_mask, w.shape)
                model.cost_reg += tf.nn.l2_loss((w - w_val) * w_mask)
            model.set_optimizer(var_list=var_list)


        for epoch in range(hp['EPOCHS']):
            step = 0
            while step * hp['batch_size_train'] < 60000:
                try:

                    # TODO: WINNIE ADD
                    hp['current_step'] = step

                    # Validation
                    if step % display_step == 0:
                        log['trials'].append(step * hp['batch_size_train'])
                        log['times'].append(time.time() - t_start)
                        log = do_eval(sess, model, log, hp['rule_trains'], epoch)
                        # if log['perf_avg'][-1] > model.hp['target_perf']:
                        # check if minimum performance is above target
                        # TODO: winnie changed

                        if rich_output:
                            display_rich_output(model, sess, step, log, model_dir)

                    # Training
                    # TODO: Winnie changed
                    if hp['rule_trains'] == ['mnist']:
                        rule_train_now = 'mnist'
                    else:
                        rule_train_now = hp['rng'].choice(hp['rule_trains'],
                                                      p=hp['rule_probs'])

                    # Generate a random batch of trials.
                    # Each batch has the same trial length
                    # TODO: WINNIE ADD, DELETE IF INCORRECT
                    if hp['rule_trains'] == ['mnist']:
                        trial_train = generate_trials(
                            rule_train_now, hp, 'train',
                            batch_size=hp['batch_size_train'])
                    else:
                        trial_train = generate_trials(
                            rule_train_now, hp, 'random',
                            batch_size=hp['batch_size_train'])

                    # Generating feed_dict.
                    feed_dict = tools.gen_feed_dict(model, trial_train, hp)

                    # TODO: winnie changed
                    _, c_lsq_train, y_hat_train = sess.run(
                        [model.train_step, model.cost_lsq, model.y_hat],
                        feed_dict=feed_dict)
                    if hp['rule_trains'] == ['mnist']:
                        perf_train = np.mean(get_perf(y_hat_train, trial_train.y, hp, rule='mnist'))

                        log['perf_train_mnist' + str(epoch)].append(perf_train)
                    step += 1

                except KeyboardInterrupt:
                    print("Optimization interrupted by user")
                    break

            log['perf_vali_avg_mnist'].append(np.mean(log['perf_vali_mnist' + str(epoch)]))
            log['perf_train_avg_mnist'].append(np.mean(log['perf_train_mnist' + str(epoch)]))

        print("Optimization finished!")




if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--modeldir', type=str, default='data/debug/mnist/20rnn_20dt_0.5stim_0.5res_end')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    hp = {'activation': 'softplus',
          # TODO: winnie changed
          'mix_rule': False,
          'l1_h': 0.,
          'use_separate_input': True,

          'EPOCHS': 1,
          'n_rnn': 20,  # 20*20
          'dt': 20,
          'stim_times': 0.5,
          'res_times': 0.5,
          'off': -1,   # when should output be turned off # hp['off'] = int((400+600 * hp['stim_times']) / hp['dt'] - 1)
          }
    train(args.modeldir,
          seed=1,
          hp=hp,
          ruleset='mnist',
          rule_trains=['contextdelaydm1', 'contextdelaydm2',
                       'contextdm1', 'contextdm2'],
          display_step=1)
