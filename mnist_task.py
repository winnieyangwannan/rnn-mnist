import numpy as np

def _mnist(config, mode, **kwargs):
    dt = config['dt']
    rng = config['rng']

    if mode == 'vali':
        batch_size = kwargs['batch_size']

        # Time of stimuluss on/off
        stim_on = int(rng.uniform(100,400)/dt)
        stim_ons = (np.ones(batch_size)*stim_on).astype(int)

        stim_dur = int(rng.choice([400, 800, 1600])/dt)
        fix_offs = (stim_ons+stim_dur).astype(int)
        # each batch consists of sequences of equal length
        tdim = stim_on+stim_dur+int(500/dt)


    elif mode == 'train':
        batch_size = kwargs['batch_size']

        # Time of stimuluss on/off
        stim_on = int(rng.uniform(100,400)/dt)
        stim_ons = (np.ones(batch_size)*stim_on).astype(int)

        stim_dur = int(rng.choice([400, 800, 1600])/dt)
        fix_offs = (stim_ons+stim_dur).astype(int)

        # each batch consists of sequences of equal length
        tdim = stim_on+stim_dur+int(500/dt)



    elif mode == 'test':
        batch_size = kwargs['batch_size_test']

        # Dense coverage of the stimulus space
        stim_on = int(500/dt)
        stim_ons = (np.ones(batch_size) * stim_on).astype(int)

        fix_off = int(2000/dt)
        stim_offs = (np.ones(batch_size) * fix_off).astype(int)

        tdim = int(2500/dt)

    # time to check the saccade location
    check_ons = fix_offs + int(100 / dt)

    trial = Trial_mnist(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1_locs, ons=stim_ons, offs=fix_offs, strengths=stim1_strengths, mods=stim_mod)
    trial.add('stim', stim2_locs, ons=stim_ons, offs=fix_offs, strengths=stim2_strengths, mods=stim_mod)
    trial.add('fix_out', offs=fix_offs)
    stim_locs = [stim1_locs[i] if (stim1_strengths[i] > stim2_strengths[i])
                 else stim2_locs[i] for i in range(batch_size)]
    trial.add('out', stim_locs, ons=fix_offs)

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
        self.pref = np.arange(0, 2 * np.pi, 2 * np.pi / self.n_eachring)  # preferences

        self.batch_size = batch_size
        self.tdim = tdim
        self.x = np.zeros((tdim, batch_size, self.n_input), dtype=self.float_type)
        self.y = np.zeros((tdim, batch_size, self.n_output), dtype=self.float_type)
        if self.config['loss_type'] == 'lsq':
            self.y[:, :, :] = 0.05
        # y_loc is the stimulus location of the output, -1 for fixation, (0,2 pi) for response
        self.y_loc = -np.ones((tdim, batch_size), dtype=self.float_type)

        self._sigma_x = config['sigma_x'] * np.sqrt(2 / config['alpha'])

    def expand(self, var):
        """Expand an int/float to list."""
        if not hasattr(var, '__iter__'):
            var = [var] * self.batch_size
        return var

    def add(self, loc_type, locs=None, ons=None, offs=None, strengths=1, mods=None):
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
        strengths = self.expand(strengths)
        mods = self.expand(mods)

        for i in range(self.batch_size):
            if loc_type == 'fix_in':
                self.x[ons[i]: offs[i], i, 0] = 1
            elif loc_type == 'stim':
                # Assuming that mods[i] starts from 1
                self.x[ons[i]: offs[i], i, 1 + (mods[i] - 1) * self.n_eachring:1 + mods[i] * self.n_eachring] \
                    += self.add_x_loc(locs[i]) * strengths[i]
            elif loc_type == 'fix_out':
                # Notice this shouldn't be set at 1, because the output is logistic and saturates at 1
                if self.config['loss_type'] == 'lsq':
                    self.y[ons[i]: offs[i], i, 0] = 0.8
                else:
                    self.y[ons[i]: offs[i], i, 0] = 1.0
            elif loc_type == 'out':
                if self.config['loss_type'] == 'lsq':
                    self.y[ons[i]: offs[i], i, 1:] += self.add_y_loc(locs[i]) * strengths[i]
                else:
                    y_tmp = self.add_y_loc(locs[i])
                    y_tmp /= np.sum(y_tmp)
                    self.y[ons[i]: offs[i], i, 1:] += y_tmp
                self.y_loc[ons[i]: offs[i], i] = locs[i]
            else:
                raise ValueError('Unknown loc_type')

    def add_x_noise(self):
        """Add input noise."""
        self.x += self.config['rng'].randn(*self.x.shape) * self._sigma_x

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

    def add_rule(self, rule, on=None, off=None, strength=1.):
        """Add rule input."""
        if isinstance(rule, int):
            self.x[on:off, :, self.config['rule_start'] + rule] = strength
        else:
            ind_rule = get_rule_index(rule, self.config)
            self.x[on:off, :, ind_rule] = strength

    def add_x_loc(self, x_loc):
        """Input activity given location."""
        dist = get_dist(x_loc - self.pref)  # periodic boundary
        dist /= np.pi / 8
        return 0.8 * np.exp(-dist ** 2 / 2)

    def add_y_loc(self, y_loc):
        """Target response given location."""
        dist = get_dist(y_loc - self.pref)  # periodic boundary
        if self.config['loss_type'] == 'lsq':
            dist /= np.pi / 8
            y = 0.8 * np.exp(-dist ** 2 / 2)
        else:
            # One-hot output
            y = np.zeros_like(dist)
            ind = np.argmin(dist)
            y[ind] = 1.
        return y


