import numpy as np
import matplotlib.pyplot  as plt
from tqdm import tqdm

from src.fmcw_simulation.simulation import FmcwRadar
from src.py_network.adaptive_resonate import AdaptiveResonate

class SpikeTester:

    def __init__(self, radar_confg, targets, random_start=True):

        self.radar_conifg = radar_confg
        self.targets = targets
        self.random_start = random_start

        self.fmcw = FmcwRadar(self.radar_conifg)
        self.fmcw.set_targets(self.targets)

        return
    
    def initialize_network(self):
        n_units = len(self.targets)
        t_max = self.fmcw.t_chirp * self.fmcw.n_chirps
        t_res = self.fmcw.t_chirp / self.fmcw.n_samples
        self.res_net = AdaptiveResonate(
                    n_units, 
                    t_max, 
                    self.fmcw.t_chirp / self.fmcw.n_samples, 
                    damping=0, 
                    thresh=0.7,
                    record_hist=False,
                    k=1,
                    feedback=True,
                    w_scale= np.ones((self.fmcw.n_rx, n_units))*t_res*100*2,
                    n_rxs=self.fmcw.n_rx
        )

        # res_net.spike_condition = res_net.period_spiking
        self.res_net.normalize_input = False
        self.res_net.normalize_neuron = True
        self.res_net.spike_threshold = self.fmcw.get_freq_from_range(0.5) / (2 * np.pi)

        for rx in range(self.fmcw.n_rx):
            if not self.random_start:
                max_range = self.fmcw.max_range / 3
                start_ranges = ((np.linspace(1, max_range, self.res_net.nfreq)))
                start_freqs = self.fmcw.get_freq_from_range(start_ranges)
                self.res_net.ws[rx] = start_freqs * 2 * np.pi
            else:
                max_freq = self.fmcw.get_freq_from_range(self.fmcw.get_ranges()[-1])
                self.res_net.ws[rx] = np.random.rand(self.res_net.nfreq) * max_freq
            self.res_net.starting_frequency = np.copy(self.res_net.ws)
                    
    def run_test(self, spike_threshold, frames_to_use=1):
        
        self.initialize_network()

        self.res_net.spike_threshold = spike_threshold

        self.ws_hist = np.zeros((self.res_net.n_steps * frames_to_use, self.res_net.n_units))
        self.spikes_out = np.zeros((self.res_net.n_steps * frames_to_use, self.res_net.n_units))
        self.vs_hist = np.zeros((self.res_net.n_steps * frames_to_use, self.res_net.n_units), dtype=complex)

        for f in range(frames_to_use):
            
            frame = self.fmcw.generate_data_snr()
            # frame = self.fmcw.apply_hann(frame)
            flat_frame = frame.reshape((frame.shape[0], -1))

            # must use C++ implementation for spike output
            vs_hist_tmp, ws_hist_tmp = self.res_net.update_neurons_c(flat_frame)
            
            self.ws_hist[f*self.res_net.n_steps:(f+1)*self.res_net.n_steps] = ws_hist_tmp.reshape((self.res_net.n_steps, self.res_net.n_units))
            self.vs_hist[f*self.res_net.n_steps:(f+1)*self.res_net.n_steps] = vs_hist_tmp.reshape((self.res_net.n_steps, self.res_net.n_units))
            self.spikes_out[f*self.res_net.n_steps:(f+1)*self.res_net.n_steps] = self.res_net.spikes_out.reshape((self.res_net.n_steps, self.res_net.n_units))

    def measure_error(self):

        # error is measured as average error compared to ws_hist
        integrated_ws = self.res_net.integrate_spikes(self.spikes_out)
        error = np.abs(self.ws_hist.squeeze() - integrated_ws.squeeze()) / (2 * np.pi)
        error = np.square(error)

        mean_sq_error = np.mean(error)
        rmse = np.sqrt(mean_sq_error)

        return rmse
    
    def measure_sparsity(self):
        
        n_steps_neurons = self.ws_hist.size
        n_spikes = self.get_number_of_spikes()

        sparsity = 1 - (n_spikes / n_steps_neurons)

        return sparsity
    
    def get_number_of_spikes(self):
        n_spikes = np.sum(np.abs(self.spikes_out))
        return n_spikes

