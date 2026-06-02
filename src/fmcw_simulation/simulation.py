import numpy as np
import matplotlib.pyplot as plt

fmcw_base_config = {
    "n_rx": 2,
    "fb" : 2.4e9,
    "B" : 750e6,
    "n_chirps": 128,
    "n_samples": 512,
    "t_chirp": 4e-5,
    "IQ": False,
    "noise_std": None,
}

class FmcwRadar:

    c = 3e8

    def __init__(self, config=fmcw_base_config):
        # import the value from a dict with the configuration info
        self.random_Ap = True    
        self.target_snr_db = None 
        
        for key, value in config.items():
            setattr(self, key, value)

        self.d_rx = (FmcwRadar.c / self.fb) / 2  
        self.sampling_freq = self.n_samples / self.t_chirp
        self.nyq = self.sampling_freq / 2
        self.slope = self.B / self.t_chirp
        self.max_range = (self.sampling_freq * FmcwRadar.c) / (self.slope * 2) 

        return
    
    def set_target_snr(self, snr):
        self.target_snr_db = snr
    
    def get_parameters(self):
        # return the parameters as a dict
        return {
            "n_rx": self.n_rx,
            "fb" : self.fb,
            "B" : self.B,
            "n_chirps": self.n_chirps,
            "n_samples": self.n_samples,
            "t_chirp": self.t_chirp,
            "IQ": self.IQ,
            "noise_std": self.noise_std,
            "snr": self.snr if self.snr is not None else np.nan
        }
    
    def set_targets(self, targets_info):
        # each targets contains:
        #   range
        #   velocity
        #   angle
        #   cross-section (just a value for amplitide for now)

        self.targets_info = targets_info

        for target in self.targets_info:
            self.target_add_random_phase_amplitude(target)

        return 
    
    def init_empty_frame(self):
        if self.IQ:
            frame = np.zeros((self.n_rx, self.n_chirps, self.n_samples), dtype=complex)
        else:
            frame = np.zeros((self.n_rx, self.n_chirps, self.n_samples))
        return frame
    
    def get_ranges(self):
        # used to ceonvert from frequency bin to range measurements
        freqs = np.fft.fftfreq(self.n_samples, self.t_chirp / self.n_samples)
        ranges = (FmcwRadar.c * self.t_chirp * freqs) / (2 * self.B) 
    
        return ranges[:self.n_samples // 2]
    
    def get_ranges2(self):
        
        sampling_rate = self.n_samples / self.t_chirp
        slope = self.B / self.t_chirp
        max_range = (sampling_rate * FmcwRadar.c) / (2 * slope)
        ranges = np.linspace(0, max_range, self.n_samples)

        return ranges
    
    def get_range_from_freq(self, freq, negative=False):
        freq = np.array(freq) 
                    
        range_tmp = (FmcwRadar.c * self.t_chirp * freq) / (2 * self.B) 
        if negative is False:
            range = np.mod(range_tmp, self.max_range)
        else:
            # negative = True is only used for converting scale for plotting
            range = range_tmp
        return range
    
    def get_freq_from_range(self, range):
        return (range * 2 * self.B) / (FmcwRadar.c * self.t_chirp)
    
    def doppler_frequency_to_velocity(self, f_d):
        lam = FmcwRadar.c / self.fb  # wavelength
        return (lam / 2.0) * f_d

    def get_velocities(self):
        # convert from freq to vels
        freqs = np.fft.fftfreq(self.n_chirps, self.t_chirp)
        vels = ((FmcwRadar.c / self.fb) * freqs) / 2

        return vels #np.fft.fftshift(vels)
    
    def get_angles(self):
        # convert from freq to angles
        # something might be wrong here
        freqs = np.fft.fftfreq(self.n_rx)
        angles = np.arcsin(freqs * 2)

        return np.fft.fftshift(angles)
    
    def target_add_random_phase_amplitude(self, target):
        assert type(target) is dict

        A_rand = np.random.rand() * 2.5 + 0.5
        p_rand = np.random.rand() * 2 * np.pi

        if self.random_Ap:
            target['A_rand'] = A_rand
            target['p_rand'] = p_rand
        else:
            target['A_rand'] = 1
            target['p_rand'] = p_rand

    def target_add_random_phase(self, target):
        assert type(target) is dict

        p_rand = np.random.rand() * 2 * np.pi

        target['p_rand'] = p_rand

    def generate_data(self):

        frame = self.init_empty_frame()
        
        if type(self.targets_info) is dict:
            self.targets_info = [self.targets_info]
        
        # generate random phase and amplitude for each target
        # they are constant withina frame
        for target in self.targets_info:
            self.target_add_random_phase_amplitude(target)

        # generate IF signal for a each target and sum them
        for target in self.targets_info:
            frame_target = self._generate_single_target(target)
            frame += frame_target

        noise_std = self.noise_std if self.noise_std is not None else 0.0
        noise = 0.0 + 0*1j
        noise = np.random.normal(0, noise_std, frame.shape)
        if self.IQ:
            noise = noise.astype(complex) + 1j*np.random.normal(0, noise_std, frame.shape)

        if not self.IQ:
            p_sign = np.abs(frame**2).mean() #/ frame.size
            p_noise = np.abs(noise**2).mean() #/ noise.size
        else:
            p_sign = np.mean(frame.imag**2) + np.mean(frame.real**2)
            p_noise = np.mean(noise.imag**2) + np.mean(noise.real**2)

        self.snr = 10*np.log10(p_sign / p_noise)
    
        self.noise = noise
        self.pure_signal = np.copy(frame)
        frame += noise
        
        return frame
    
    def generate_data_snr(self):
        frame = self.init_empty_frame()
            
        
        # ... (target generation logic) ...
        for target in self.targets_info:
            # self.target_add_random_phase_amplitude(target)
            self.target_add_random_phase(target)
            frame += self._generate_single_target(target)

        # Calculate Signal Power (Ps)
        p_signal = np.mean(np.abs(frame)**2)

        if self.target_snr_db is not None:
            # Calculate required noise power based on desired SNR
            # SNR = 10 * log10(Ps / Pn) -> Pn = Ps / 10^(SNR/10)
            p_noise = p_signal / (10**(self.target_snr_db / 10))
            noise_std = np.sqrt(p_noise)
        else:
            noise_std = self.noise_std if self.noise_std is not None else 0.0

        # Generate Noise
        if self.IQ:
            # Split variance between I and Q
            std_per_component = noise_std / np.sqrt(2)
            noise = (np.random.normal(0, std_per_component, frame.shape) + 
                    1j * np.random.normal(0, std_per_component, frame.shape))
        else:
            noise = np.random.normal(0, noise_std, frame.shape)

        self.snr = 10 * np.log10(p_signal / np.mean(np.abs(noise)**2))
        self.pure_signal = np.copy(frame)
        
        return frame + noise
    
    def _generate_single_target(self, target):
        frame = self.init_empty_frame()

        for chirp_id in range(self.n_chirps):
            for rx_id in range(self.n_rx):
                frame[rx_id, chirp_id] = self._generate_single_target_chirp(target, chirp_id, rx_id)
            # target['range'] += target["velocity"] * self.t_chirp

        return frame
        
    # def _generate_single_target_chirp(self, target, chirp_id, rx_id):
    #     t = np.linspace(0, self.t_chirp,
    #                     self.n_samples, endpoint=False)
    #     # f_beat = self._compute_f_beat(target['range'])
    #     f_beat = self._compute_f_beat(target['range'] + target['velocity']*self.t_chirp*chirp_id)
    #     phase_diff_vel = chirp_id * self._compute_phase_diff_vel(target['velocity'])
    #     phase_diff_angle = (rx_id+1) * self._compute_phase_diff_angle(target['angle'])
        
    #     beat_signal = target['A_rand'] * \
    #                   np.exp(1j*(2 * np.pi * f_beat * t \
    #                             + phase_diff_vel \
    #                             + phase_diff_angle \
    #                             + target['p_rand']))

    #     if not self.IQ:
    #         return beat_signal.real
        
    #     return beat_signal
    
    def _generate_single_target_chirp(self, target, chirp_id, rx_id):

        t = np.linspace(0, self.t_chirp, self.n_samples, endpoint=False)
        
        # Current range at the start of this specific chirp
        # Range changes linearly over the frame
        current_range = target['range'] #+ target['velocity'] * (chirp_id * self.t_chirp)
        
        # Physics-based FMCW beat signal: 
        # 1. Beat freq due to slope (S * tau)
        # 2. Phase shift due to R (2 * pi * fc * tau)
        # tau = 2 * R / c
        
        slope = self.B / self.t_chirp
        tau = 2 * current_range / self.c
        fc = self.fb
        
        # The term 2*pi*slope*tau*t handles the beat frequency
        # The term 2*pi*fc*tau handles the phase (Doppler across chirps)
        # beat_phase = 2 * np.pi * (slope * tau * t + fc * tau)
        lambda_c = self.c / fc
        f_d = 2 * target['velocity'] / lambda_c

        phase_doppler = 2 * np.pi * f_d * chirp_id * self.t_chirp

        beat_phase = 2 * np.pi * (slope * tau * t) + phase_doppler

        # Angle phase shift (Spatial frequency)
        # Using rx_id (0, 1, ...) without the +1
        phase_angle = 0 #2 * np.pi * (self.d_rx * np.sin(target['angle']) / (self.c / fc)) * rx_id
        
        beat_signal = target['A_rand'] * np.exp(1j * (beat_phase + phase_angle + target['p_rand']))

        return beat_signal if self.IQ else beat_signal.real

    def _compute_f_beat(self, R):
        return (2 * R * self.B) / (FmcwRadar.c * self.t_chirp)

    def _compute_phase_diff_vel(self, v):
        return self._compute_phase_diff(2 * v * self.t_chirp)
    
    def _compute_phase_diff_angle(self, angle):
        return self._compute_phase_diff(self.d_rx * np.sin(angle)) 
    
    def _compute_phase_diff(self, delta_d):
        delta_time = delta_d / FmcwRadar.c
        return 2 * np.pi * self.fb * delta_time
    
    def apply_hann(self, frame):

        new_frame = np.zeros_like(frame)
    
        hann_w = np.hanning(self.n_samples)[None, None, :]

        # remove DC per chirp
        frame = frame - frame.mean(axis=2, keepdims=True)

        # apply window
        new_frame = frame * hann_w

        return new_frame
        
