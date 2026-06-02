import numpy as np
import mat73
import h5py

from scipy.ndimage import maximum_filter

from types import SimpleNamespace

def extract_targets(
    rd_map,
    ranges,
    cels,
    guard_r=2,
    guard_d=2,
    train_r=8,
    train_d=4,
    pfa=1e-5,
    min_distance=3,
):
    """
    Extract target ranges and cels from RD map.
    """

    detections, threshold_map = ca_cfar_2d(
        rd_map,
        guard_r=guard_r,
        guard_d=guard_d,
        train_r=train_r,
        train_d=train_d,
        pfa=pfa,
    )

    magnitude = np.abs(rd_map)

    # keep only local maxima
    local_max = maximum_filter(
        magnitude,
        size=min_distance,
    ) == magnitude

    peaks = detections & local_max

    peak_indices = np.argwhere(peaks)

    targets = []

    for r_bin, d_bin in peak_indices:

        target = {
            "range_bin": int(r_bin),
            "doppler_bin": int(d_bin),
            "range_m": ranges[r_bin],
            "velocity_m_s": cels[d_bin],
            "magnitude": magnitude[r_bin, d_bin],
        }

        targets.append(target)

    # sort by magnitude
    targets = sorted(
        targets,
        key=lambda x: x["magnitude"],
        reverse=True,
    )

    return targets, peaks, threshold_map

def ca_cfar_2d(
    rd_map,
    guard_r=2,
    guard_d=2,
    train_r=8,
    train_d=4,
    pfa=1e-5,
):
    """
    2D Cell-Averaging CFAR for Range-Doppler maps.

    Parameters
    ----------
    rd_map : ndarray
        2D complex or magnitude RD map
        shape = [n_range_bins, n_doppler_bins]

    guard_r : int
        Guard cells in range dimension

    guard_d : int
        Guard cells in doppler dimension

    train_r : int
        Training cells in range dimension

    train_d : int
        Training cells in doppler dimension

    pfa : float
        Probability of false alarm

    Returns
    -------
    detections : ndarray(bool)
        CFAR detection mask

    threshold_map : ndarray
        Adaptive threshold map
    """

    rd_power = np.abs(rd_map) ** 2

    nr, nd = rd_power.shape

    detections = np.zeros_like(rd_power, dtype=bool)
    threshold_map = np.zeros_like(rd_power)

    # total window size
    win_r = train_r + guard_r
    win_d = train_d + guard_d

    # number of training cells
    n_train = (
        (2 * win_r + 1) * (2 * win_d + 1)
        - (2 * guard_r + 1) * (2 * guard_d + 1)
    )

    # CA-CFAR scaling factor
    alpha = n_train * (pfa ** (-1 / n_train) - 1)

    for r in range(win_r, nr - win_r):
        for d in range(win_d, nd - win_d):

            r0 = r - win_r
            r1 = r + win_r + 1

            d0 = d - win_d
            d1 = d + win_d + 1

            window = rd_power[r0:r1, d0:d1].copy()

            # remove guard cells + CUT
            g0r = win_r - guard_r
            g1r = win_r + guard_r + 1

            g0d = win_d - guard_d
            g1d = win_d + guard_d + 1

            window[g0r:g1r, g0d:g1d] = 0

            noise_est = np.sum(window) / n_train

            threshold = alpha * noise_est

            threshold_map[r, d] = threshold

            if rd_power[r, d] > threshold:
                detections[r, d] = True

    return detections, threshold_map

def load_data_cascaded(path_to_folder):
    frames = []
    for i in range(1, 9): #TODO read number of frames

        dataPath = f'{path_to_folder}/{i}.mat'
        
        data = mat73.loadmat(dataPath)
        adcData = data['adcData']
        
        frames.append(adcData[:, :, 0, 0].T)

    # Concatenate along chirp dimension
    adcData_all = np.concatenate(frames, axis=0)
    
    return adcData_all

import numpy as np

def load_and_build_radar_cube(file_path):

    f = h5py.File(file_path, 'r')
    real = f['real']
    imag = f['imag']

    data = np.array(real) + 1j * np.array(imag)

    return data
    

def get_radar_parameters(fs_select):
    Param = SimpleNamespace()
    # --- Radar System Parameter Configuration ---
    Param.ntt_range   = 256;      # Number of range bins (ADC samples per chirp)
    Param.ntt_azimuth = 128;      # Number of Doppler bins (Chirps per frame)
    Param.idle_time   = 5*1e-6;   # Idle time between chirps
    Param.ramp_start  = 6*1e-6;   # ADC start time within the ramp
    Param.ramp_end    = 50*1e-6;  # End time of the frequency ramp

    # Total time for a sequence of chirps (likely a burst of 12)
    Param.T           = (Param.ramp_end + Param.idle_time) * 12; 
    Param.mu          = 35;       # Slope constant (frequency slope in MHz/us or similar unit)
    Param.PRF         = 1 / (Param.T); # Pulse Repetition Frequency
    Param.Periodicity = (Param.T) * 12 * Param.ntt_azimuth;

    fs_choice = [0.6e7, 1.2e7, 1.8e7]
    Param.fs          = fs_choice[fs_select];    # Sampling rate (18 MHz) #other two measurement1.2e7 0.6e7 1.8e7
    Param.Ts          = 1 / Param.fs;
    Param.T_d         = Param.ntt_range * Param.Ts; # Effective sampling duration

    # Bandwidth calculation (B = slope * duration)
    Param.B           = Param.mu * Param.T_d * 1e12; 
    Param.c           = 3e8;      # Speed of light
    Param.resolution  = Param.c / (2 * Param.B); # Range resolution

    Param.f0          = 76e9;     # Start frequency (76 GHz)
    Param.fc          = Param.f0 + Param.B/2; # Center frequency
    Param.lambda_val      = Param.c / Param.fc;   # Wavelength
    Param.dr          = Param.lambda_val / 2;     # Element spacing (Half-wavelength)
    Param.tm          = Param.T_d;
    Param.Ne          = 6;        # Number of Elevation elements (likely)
    Param.Na          = 86;       # Number of Azimuth elements (Total Virtual Channels)
    snr               = 20; 

    return Param 


def get_radar_parameters2():
    Param = SimpleNamespace()
    
    # --- Radar System Parameter Configuration ---
    Param.ntt_range   = 128      # Number of range bins (ADC samples per chirp)
    Param.ntt_azimuth = 128      # Number of Doppler bins (Chirps per frame)
    Param.idle_time   = 20 * 1e-6 # Idle time between chirps (20 us)
    Param.ramp_start  = 0 * 1e-6  # Assumed 0 us start time based on given script
    Param.ramp_end    = 400 * 1e-6 # End time of the frequency ramp (400 us)

    # Total time for a sequence of chirps per frame 
    # (chirps_per_frame = 128 based on (128 * 2) / 2)
    Param.T           = (Param.ramp_end + Param.idle_time) * 128 
    Param.mu          = 10        # Slope constant (frequency slope in MHz/us)
    Param.PRF         = 1 / (Param.ramp_end + Param.idle_time) # Pulse Repetition Frequency (1/tm)
    Param.Periodicity = (Param.ramp_end + Param.idle_time) * 128 # Frame period duration

    # Sampling rate options from your experiment setup
    fs_choice         = [2.95e6, 5.95e6, 10.95e6]
    fs_select         = 0         # Selecting 2.95e6 by default
    Param.fs          = fs_choice[fs_select]    
    Param.Ts          = 1 / Param.fs
    Param.T_d         = Param.ntt_range * Param.Ts # Effective sampling duration

    # Bandwidth calculation (B = slope * duration in us)
    Param.B           = Param.mu * (Param.ramp_end * 1e12) # Resulting in Hz
    Param.c           = 3e8       # Speed of light [m/s]
    Param.resolution  = Param.c / (2 * (Param.B * 1e6)) # Range resolution using true Hz Bandwidth

    Param.f0          = 60e9      # Start frequency (60 GHz)
    Param.fc          = Param.f0 + (Param.B * 1e6) / 2 # Center frequency [Hz]
    Param.lambda_val  = Param.c / Param.f0   # Wavelength using start frequency (per your MATLAB setup)
    Param.dr          = Param.lambda_val / 2 # Element spacing (Half-wavelength)
    Param.tm          = Param.ramp_end + Param.idle_time # Total chirp time duration
    Param.Ne          = 2         # Swapped to match #TX from your script (Transmitters acting as elevation/MIMO keys)
    Param.Na          = 8         # Swapped to match Total RX channels / Virtual Channels from your script
    snr               = 20        # Signal-to-noise ratio base configuration

    return Param