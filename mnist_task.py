import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def load_mnist_data():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0

    #x_vali = x_test[0: int(len(x_test)*0.6)]
    #y_vali = y_test[0: int(len(y_test)*0.6)]

    #x_test = x_test[int(len(x_test) * 0.4):]
    #y_test = y_test[int(len(y_test) * 0.4):]

    # flatten the image
    x_train = x_train.reshape(len(x_train), -1)
    #x_vali = x_vali.reshape(len(x_vali), -1)
    x_test = x_test.reshape(len(x_test), -1)


    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = load_mnist_data()
a = 1

def filter_digit(x, y, filter_digit):
    keep = (y == filter_digit)
    x, _ = x[keep], y[keep]
    y = np.where(y == filter_digit)[0]
    return x, y


#x_train_3, y_train_3 = filter_digit(x_train, y_train, 3)
#a = 1
#x_train_3 = x_train_3.reshape(len(x_train_3), 28, 28, 1)
#plt.imshow(x_train_3[0, :, :, 0])
#plt.colorbar()
#plt.show()
#a=1


def _mnist(config, mode, **kwargs):

    dt = config['dt']
    rng = config['rng']
    step = config['current_step']
    times = config['times']

    # load mnist
    x_train, x_test, y_train, y_test = load_mnist_data()


    if mode == 'vali':
        batch_size = config['batch_size_vali']
        y_test = tf.keras.utils.to_categorical(y_test)

        # Time of stimuluss on/off
        stim_on = int(400*times/dt)
        stim_ons = (np.ones(batch_size)*stim_on).astype(int)

        stim_dur = int(rng.choice([600*times])/dt)
        fix_offs = (stim_ons+stim_dur).astype(int)
        # each batch consists of sequences of equal length
        tdim = stim_on+stim_dur+int(500*times/dt)

        stim_batch = x_test[step*batch_size:(step+1)*batch_size, :]
        target_batch = y_test[step*batch_size:(step+1)*batch_size, :]


    elif mode == 'train':
        # TODO: change test length here later
        batch_size = config['batch_size_train']
        y_train = tf.keras.utils.to_categorical(y_train)

        # Time of stimuluss on/off
        stim_on = int(400*times/dt)
        stim_ons = (np.ones(batch_size)*stim_on).astype(int)

        stim_dur = int(rng.choice([600*times])/dt)
        fix_offs = (stim_ons+stim_dur).astype(int)
        # each batch consists of sequences of equal length
        tdim = stim_on+stim_dur+int(500*times/dt)

        stim_batch = x_train[step*batch_size:(step+1)*batch_size, :]
        target_batch = y_train[step*batch_size:(step+1)*batch_size, :]



    elif mode == 'test':
        # TODO: things need to be fixed here
        batch_size = config['batch_size_test']
        y_test = tf.keras.utils.to_categorical(y_test)

        # Time of stimuluss on/off
        stim_on = int(400*times/dt)
        stim_ons = (np.ones(batch_size)*stim_on).astype(int)

        stim_dur = int(rng.choice([600*times])/dt)
        fix_offs = (stim_ons+stim_dur).astype(int)
        # each batch consists of sequences of equal length
        tdim = stim_on+stim_dur+int(500*times/dt)

        stim_batch = x_test[step*batch_size:(step+1)*batch_size, :]
        target_batch = y_test[step*batch_size:(step+1)*batch_size, :]


    # time to check the saccade location
    # TODO: check what does this do
    check_ons = fix_offs + int(100 / dt)


    trial = Trial_mnist(config, tdim, batch_size)
    #trial.add('fix_in', stims=stim_batch, targets=target_batch, ons=stim_ons, offs=fix_offs)
    trial.add('stim',  stims=stim_batch, targets=target_batch, ons=stim_ons, offs=fix_offs)
    #trial.add('fix_out', stims=stim_batch, targets=target_batch, ons=stim_ons, offs=fix_offs)
    trial.add('out', stims=stim_batch, targets=target_batch, ons=stim_ons, offs=fix_offs)

    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {'fix1': (None, stim_ons),
                    'stim1': (stim_ons, fix_offs),
                    'go1': (fix_offs, None)}

    return trial




class Trial_mnist(object):
    """Class representing a batch of trials."""

    def __init__(self, config, tdim, batch_size):
        """A batch of trials.

        Args:
            config: dictionary of configurations
            tdim: int, number of time steps
            batch_size: int, batch size
        """
        self.float_type = 'float32'  # This should be the default
        self.config = config
        self.dt = self.config['dt']


        self.n_eachring = self.config['n_eachring']
        self.n_input = self.config['n_input']
        self.n_output = self.config['n_output']

        self.batch_size = int(batch_size)
        self.tdim = int(tdim)
        self.x = np.zeros((self.tdim, batch_size, self.n_input), dtype=self.float_type)
        self.y = np.zeros((self.tdim, batch_size, self.n_output), dtype=self.float_type)

        if self.config['loss_type'] == 'lsq':
            self.y[:, :, :] = 0.05

        # y_loc is the stimulus location of the output, -1 for fixation, (0,2 pi) for response
        self.y_loc = -np.ones((self.tdim, batch_size), dtype=self.float_type)
        # TODO: winnie comment out
        #self._sigma_x = config['sigma_x'] * np.sqrt(2 / config['alpha'])

    def expand(self, var):
        """Expand an int/float to list."""
        if not hasattr(var, '__iter__'):
            var = [var] * self.batch_size
        return var

    def add(self, loc_type, stims=None, targets=None, ons=None, offs=None, mods=None):
        """Add an input or stimulus output.

        Args:
            loc_type: str (fix_in, stim, fix_out, out), type of information to be added
            locs: array of list of float (batch_size,), locations to be added, only for loc_type=stim or out
            ons: int or list, index of onset time
            offs: int or list, index of offset time
            strengths: float or list, strength of input or target output
            mods: int or list, modalities of input or target output
        """

        ons = self.expand(ons)
        offs = self.expand(offs)


        for i in range(self.batch_size):
            #if loc_type == 'fix_in':
                #self.x[ons[i]: offs[i], i, 0] = 1
            if loc_type == 'stim':
                # Assuming that mods[i] starts from 1
                stim = stims[i, :]
                stim = np.tile(stim, (offs[i]-ons[i], 1))  # tile to the length of the time
                self.x[ons[i]: offs[i], i, :] += stim
            #elif loc_type == 'fix_out':
                # Notice this shouldn't be set at 1, because the output is logistic and saturates at 1
             #   if self.config['loss_type'] == 'lsq':
              #      self.y[ons[i]: offs[i], i, 0] = 0.8
               # else:
                #    self.y[ons[i]: offs[i], i, 0] = 1.0
            elif loc_type == 'out':
                if self.config['loss_type'] == 'lsq':
                    target = targets[i, :] # curreng batch
                    if self.config['off'] == -1:
                        target = np.tile(target, (len(self.y) - offs[i], 1))
                        self.y[offs[i]:, i, :] += target
                    else:
                        target = np.tile(target, (offs[i]-ons[i], 1))
                        self.y[ons[i]:offs[i], i, :] += target

            else:
                raise ValueError('Unknown loc_type')

    #def add_x_noise(self):
     #   """Add input noise."""
      #  self.x += self.config['rng'].randn(*self.x.shape) * self._sigma_x

    def add_c_mask(self, pre_offs, post_ons):
        """Add a cost mask.

        Usually there are two periods, pre and post response
        Scale the mask weight for the post period so in total it's as important
        as the pre period
        """

        pre_on = int(100 / self.dt)  # never check the first 100ms
        pre_offs = self.expand(pre_offs)
        post_ons = self.expand(post_ons)

        if self.config['loss_type'] == 'lsq':
            c_mask = np.zeros((self.tdim, self.batch_size, self.n_output), dtype=self.float_type)
            for i in range(self.batch_size):
                # Post response periods usually have the same length across tasks
                c_mask[post_ons[i]:, i, :] = 5.
                # Pre-response periods usually have different lengths across tasks
                # To keep cost comparable across tasks
                # Scale the cost mask of the pre-response period by a factor
                c_mask[pre_on:pre_offs[i], i, :] = 1.

            # self.c_mask[:, :, 0] *= self.n_eachring # Fixation is important
            c_mask[:, :, 0] *= 2.  # Fixation is important

            self.c_mask = c_mask.reshape((self.tdim * self.batch_size, self.n_output))
        else:
            c_mask = np.zeros((self.tdim, self.batch_size), dtype=self.float_type)
            for i in range(self.batch_size):
                # Post response periods usually have the same length across tasks
                # Having it larger than 1 encourages the network to achieve higher performance
                c_mask[post_ons[i]:, i] = 5.
                # Pre-response periods usually have different lengths across tasks
                # To keep cost comparable across tasks
                # Scale the cost mask of the pre-response period by a factor
                c_mask[pre_on:pre_offs[i], i] = 1.

            self.c_mask = c_mask.reshape((self.tdim * self.batch_size,))
            self.c_mask /= self.c_mask.mean()

    #def add_rule(self, on=None, off=None, strength=1.):
     #   """Add rule input."""
      #  self.x[on:off, :, -1] = strength


