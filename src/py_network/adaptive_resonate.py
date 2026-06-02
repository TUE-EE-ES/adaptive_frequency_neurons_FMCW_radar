from numba import jit
import numpy as np
from . import adaptive_oscillator

class AdaptiveResonate:

    def __init__(self, n_units, sim_time, t_res, k, feedback, w_scale, n_rxs=1):
        
        self.t_res = t_res
        self.sim_time = sim_time
        self.n_steps = int(sim_time / t_res)
        self.nfreq = n_units
        self.n_units = n_units
        self.n_rxs = n_rxs
        self.k = k
        self.feedback = feedback
        self.w_scale = w_scale
        self.default_w_scale = self.w_scale.flatten()[0]

        self.total_neurons = self.n_rxs * self.nfreq

        self.normalize_input = False
        self.normalize_neuron = True
        self.wdot_mode = False
        # w_dot=True -> the value used for calculating wdot does not include the feedback

        self.spike_threshold = 0.0
        # DO NOT MULTIPLY THRESHOLD BY 2 * PI
        # it is already done in the c code

        self.ws = np.zeros((n_rxs, self.nfreq))
        self.vs = np.zeros((n_rxs, n_units), dtype=complex)

        self.ws[:] = (np.linspace(1, 10, self.nfreq)) * 2 * np.pi

        self.starting_frequency = np.copy(self.ws)

    def set_starting_frequency(self, new_starting_freqs):
        for rx in range(self.n_rxs):
            self.ws[rx] = new_starting_freqs
        self.starting_frequency = np.copy(self.ws)
    
    def get_parameters(self):
        dict_parameters = {
            't_res': self.t_res,
            'n_units': self.nfreq,
            'w_scale': self.w_scale,
            'feedback': self.feedback,
            'k': self.k,
            'n_rxs': self.n_rxs,
            'normalize_input': self.normalize_input,
            'normalize_neuron': self.normalize_neuron,
            'starting_frequencies': self.starting_frequency,
            'spike_threshold': self.spike_threshold
        }
        return dict_parameters
    
    def integrate_spikes(self, spikes=None):
        if spikes is None:
            spikes = self.spikes_out
        integrated_ws = np.cumsum(spikes * self.spike_threshold * 2 * np.pi, axis=0)
        integrated_ws += self.starting_frequency
        return integrated_ws
    
    def update_neurons(self, input_current):
        
        feedback = self.vs[0, :].sum()
        in_vals = input_current - feedback
        in_vals = in_vals / self.nfreq
        for k in range(self.nfreq):
            vs_val = self.vs[0, k] / np.abs(self.vs[0, k])
            w_dot = vs_val.imag * in_vals.real - vs_val.real * in_vals.imag
            w_dot = np.nan_to_num(w_dot)

            self.ws[0, k] -= (w_dot / self.w_scale[0, k]).reshape(self.ws[0, k].shape)
            self.vs[0, k] += in_vals.reshape(self.vs[0, k].shape)
            vs_inter = np.copy(self.vs[0, k])
            self.vs[0, k] *= np.exp(1j*self.ws[0, k]*self.t_res)
        
        return self.ws, self.vs, vs_inter
    
    def update_neurons_frame(self, in_list, intermediate=False):
        # does not support multiple rxs 
        if intermediate:
            vs_hist = np.zeros((in_list.shape[1]*2, self.vs.shape[0], self.vs.shape[1]), dtype=complex)
        else:
            vs_hist = np.zeros((in_list.shape[1], self.vs.shape[0], self.vs.shape[1]), dtype=complex)

        ws_hist = np.zeros((in_list.shape[1], self.vs.shape[0], self.vs.shape[1]))
        
        for i in range(in_list.shape[1]):

            ws, vs, vs_inter = self.update_neurons(in_list[:, i])

            if intermediate:
                vs_hist[i*2, 0] = vs_inter
                vs_hist[i*2+1, 0] = self.vs
            else:
                vs_hist[i, 0] = self.vs
            
            ws_hist[i, 0] = self.ws

        return vs_hist, ws_hist
    
    def update_neurons_c(self, in_list):
        
        # if n_rxs != 1, run multiple networks
        ws_hist_total = np.zeros((in_list.shape[1], self.n_rxs, self.n_units))
        vs_hist_total = np.zeros((in_list.shape[1], self.n_rxs, self.n_units), dtype=complex)
        spikes_out_total = np.zeros((in_list.shape[1], self.n_rxs, self.n_units), dtype=np.int8)
        
        if not np.issubdtype(in_list.dtype, np.complexfloating):
            new_in_list = np.empty_like(in_list, dtype=complex)
            new_in_list.real = in_list.real
            new_in_list.imag[:] = 0.0
            in_list = new_in_list
        
        for i in range(self.n_rxs):
            res_net_c = adaptive_oscillator.AdaptiveOscillator(
                                                n_units=self.n_units,
                                                dt=self.t_res,
                                                lambda_scale=self.w_scale[i].flatten(),
                                                alpha_lpf=1,
                                                F_input=self.wdot_mode,
                                                normalize_vs=self.normalize_neuron,
                                                normalize_input=self.normalize_input)

            res_net_c.set_ws(self.ws[i].reshape((self.n_units,)))
            res_net_c.set_vs(self.vs[i].reshape((self.n_units,)))
            res_net_c.set_spike_threshold(self.spike_threshold)
            # DO NOT MULTIPLY THRESHOLD BY 2 * PI
            # it is already done in the c code
            res_net_c.updateSequence(in_list[i].squeeze())

            vs_hist = res_net_c.get_vs_hist().reshape((-1, self.n_units))
            ws_hist = res_net_c.get_ws_hist().reshape((-1, self.n_units))

            self.ws[i] = ws_hist[-1]
            self.vs[i] = vs_hist[-1]

            ws_hist_total[:, i] = ws_hist
            vs_hist_total[:, i] = vs_hist
            spikes_out_total[:, i] = res_net_c.get_spikes_out().reshape((-1, self.n_units))
            self.spikes_out = spikes_out_total

        return vs_hist_total, ws_hist_total