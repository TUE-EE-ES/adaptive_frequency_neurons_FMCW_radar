import os, sys

import sys
sys.path.append("../../")
import datetime

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from ruamel.yaml import YAML
import time

import multiprocessing
from multiprocessing import Pool

from src.fmcw_simulation.simulation import FmcwRadar
from src.py_network.adaptive_resonate import AdaptiveResonate
from src.batch_experiments.utils import errors_2_targets, errors_n_targets
from types import SimpleNamespace

plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)
# mpl.rcParams['figure.dpi'] = 200 
mpl.rcParams['figure.figsize'] = (8, 6) 
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['legend.fontsize'] = 10
# mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#003052", "#C71818", "#18C7C7", "#70C718", "#7018C7"])
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#6ba0a6", "#d8584e", "#c09651", "#f8c08a", "#8e4f39"])

def dict_to_namespace(data):
    if type(data) is list:
        return list(map(dict_to_namespace, data))
    elif type(data) is dict:
        sns = SimpleNamespace()
        for key, value in data.items():
            setattr(sns, key, dict_to_namespace(value))
        return sns
    else:
        return data

def load_experiment_config(config_path):
    with open(config_path) as f:
        yaml = YAML(typ="safe")
        return yaml.load(f)


# initialize network
def initialize_network(
    fmcw,
    n_beams,
    random_start_frequency,
    max_start_frequency,
    n_units
):
    t_max = fmcw.t_chirp * fmcw.n_chirps
    t_res = fmcw.t_chirp / fmcw.n_samples
    res_net = AdaptiveResonate(
        n_units,
        t_max,
        fmcw.t_chirp / fmcw.n_samples,
        k=1,
        feedback=True,
        w_scale=np.ones((n_beams, n_units)) * t_res * 100 * 0.3,
        n_rxs=n_beams,
    )

    if random_start_frequency:
        res_net.ws = (
            np.random.rand(res_net.n_rxs, res_net.nfreq) * (max_start_frequency / 2) * 2 * np.pi
        )
        ## temporary
        # base = max_start_frequency * 2 * np.pi * 0.5
        # res_net.ws[:, 0] = base + (np.random.rand(res_net.n_rxs)-0.5) * ((max_start_frequency * 2 * np.pi) / 200)
        # res_net.ws[:, 1] = base + (np.random.rand(res_net.n_rxs)-0.5) * ((max_start_frequency * 2 * np.pi) / 200)
        # res_net.starting_frequency = np.copy(res_net.ws)
    else:
        base = (2 * np.pi) / fmcw.t_chirp
        for rx in range(n_beams):
            res_net.ws[rx] = np.ones(res_net.nfreq) * base

    return res_net

def run_sim(config_dict):
    cfg = dict_to_namespace(config_dict)

    fmcw = FmcwRadar(config=config_dict["radar_config"])
    wavelength = 3e8 / cfg.radar_config.fb
    sampling_rate = fmcw.n_samples / fmcw.t_chirp
    if fmcw.IQ:
        # subtract a small amount to avoid the range to circle back to zero
        max_start_freq = sampling_rate - 1e4 
    else:
        max_start_freq = sampling_rate / 2

    # ---------------- Beamforming ----------------
    if cfg.beamforming.enabled:
        d_ant = wavelength / 2
        n_beams = 129
        n_tx_virtual = cfg.radar_config.n_rx

        beamforming_angles = np.radians(np.linspace(-90, 90, n_beams))
        steering_vectors = np.exp(
            -1j
            * 2
            * np.pi
            * d_ant
            / wavelength
            * np.arange(n_tx_virtual).reshape(-1, 1)
            * np.sin(beamforming_angles)
        )
    else:
        n_beams = cfg.radar_config.n_rx
        n_tx_virtual = cfg.radar_config.n_rx

    # ---------------- Targets ----------------
    if cfg.targets.range_m.max is None:
        max_range = fmcw.get_range_from_freq(max_start_freq)
    else:
        max_range = cfg.targets.range_m.max
    # possible_ranges = np.linspace(
    #     cfg.targets.range_m.min, max_range, 10000
    # )
    if cfg.targets.range_m.min is None:
        min_range = fmcw.get_range_from_freq(0.1)
    else:
        min_range = cfg.targets.range_m.min
    
    # n_targets = rng.integers(1, cfg.targets.max_number + 1)
    n_targets = cfg.targets.max_number
    
    targets = [
        {
            "range": rng.random() * (max_range - min_range) + min_range,
            "velocity": 0.0,
            "angle": 0.0,
        }
        for i in range(n_targets)
    ]

    if hasattr(cfg.targets, "fixed_ranges"):
        fixed_ranges = np.array(cfg.targets.fixed_ranges)
        assert fixed_ranges.size == n_targets
        for i, t in enumerate(targets):
            t['range'] = fixed_ranges[i]

    # temporary
    # targets = [
    #     {
    #         "range": 0.5,
    #         "velocity": 0.0,
    #         "angle": 0.0,
    #     },
    #     {
    #         "range": 1.5,
    #         "velocity": 0.0,
    #         "angle": 0.0,
    #     }
    # ]
    if hasattr(cfg.radar_config, "noise_snr"):
        fmcw.set_target_snr(cfg.radar_config.noise_snr)

    if hasattr(cfg.radar_config, "random_ap"):
        fmcw.random_Ap = cfg.radar_config.random_ap

    if cfg.noise.randomize:
        fmcw.noise_std = (
            rng.random() * (cfg.noise.std_range.max - cfg.noise.std_range.min)
            + cfg.noise.std_range.min
        )
        if hasattr(cfg.noise, "snr_range"):
            fmcw.set_target_snr(rng.random() * (cfg.noise.snr_range.max - cfg.noise.snr_range.min)
            + cfg.noise.snr_range.min)

    fmcw.set_targets(targets)

    # ---------------- Network ----------------
    res_net = initialize_network(
        fmcw,
        n_beams=n_beams,
        random_start_frequency=cfg.network.random_start_frequency,
        max_start_frequency=max_start_freq,
        n_units=n_targets
    )

    res_net.spike_threshold = fmcw.get_freq_from_range(0.01)

    if cfg.network.random_w_scale:
        lambda_min = 1 / cfg.network.w_scale_range.max
        lambda_max = 1 / cfg.network.w_scale_range.min
        lambda_val = (
            rng.random() * (lambda_max - lambda_min)
            + lambda_min
        )
        w_scale = 1 / lambda_val
    else:
        w_scale = cfg.network.w_scale
    res_net.w_scale[:] = res_net.t_res * 100 * w_scale

    if cfg.network.random_input_normalization:
        res_net.normalize_input = rng.choice(a=[True, False])
    else:
        res_net.normalize_input = cfg.network.input_normalization

    res_net.normalize_neuron = cfg.network.amplitude_normalization

    if hasattr(cfg.network, "alpha_lpf"):
        res_net.alpha_lpf = cfg.network.alpha_lpf
    if hasattr(cfg.network, "wdot_mode"):
        res_net.wdot_mode = cfg.network.wdot_mode

    # temporary
    # fmcw.random_Ap = False
    if cfg.experiment.show_plot:
        ws_full = np.zeros((cfg.experiment.frames_per_trial,
                            fmcw.n_chirps * fmcw.n_samples, 1, res_net.n_units))

    # ---------------- Simulation loop ----------------
    for i in range(cfg.experiment.frames_per_trial):
        radar_frame = fmcw.generate_data_snr()
        if hasattr(cfg.radar_config, "hann_window") and cfg.radar_config.hann_window:
            radar_frame = fmcw.apply_hann(radar_frame)
        radar_frame_flat = radar_frame.reshape((n_tx_virtual, -1))

        if cfg.beamforming.enabled:
            flat_frame_beams = steering_vectors.T @ radar_frame_flat
        else:
            flat_frame_beams = radar_frame_flat

        ts = time.time()
        vs, ws = res_net.update_neurons_c(flat_frame_beams)
        if cfg.experiment.show_plot:
            ws_full[i] = ws
        te = time.time()
        osc_time = te - ts

    # ---------------- Results ----------------

    # account for negative frequencies
    oscillator_freqs = ws / (2 * np.pi)
    oscillator_freqs = np.mod(oscillator_freqs, fmcw.get_freq_from_range(fmcw.max_range))

    oscillator_ranges = fmcw.get_range_from_freq(oscillator_freqs) 
    oscillator_ranges_no_mod = fmcw.get_range_from_freq(ws / (2 * np.pi))    

    N = fmcw.n_samples #* 8
    ts = time.time()
    np_range_fft = (
        np.abs(np.fft.fft(radar_frame, axis=2, n=N))
        .sum(axis=1)
        .squeeze()[:]
    )
    te = time.time()
    fft_time = te - ts
    fft_peak_id = np.argsort(np_range_fft)[-n_targets:]
    fft_freqs = np.fft.fftfreq(N, fmcw.t_chirp / fmcw.n_samples)

    fft_range_detected = fmcw.get_range_from_freq(fft_freqs[fft_peak_id])

    #----------------- Convergence speed -----------------
    # this assumes only one oscillator for now
    if cfg.targets.max_number == 1:
        range_bins = fmcw.get_ranges()
        w_threshold = (range_bins[1] - range_bins[0]) / 2 # half bin size
        range_error = np.abs(oscillator_ranges.squeeze() - targets[0]['range'])
        mask = range_error < w_threshold
        convergence_time = mask.size - (mask[::-1]).argmin()
    else:
        convergence_time = None

    result_dict = {
            "range": oscillator_ranges.squeeze()[-100:].mean(axis=0),
            "frequency": oscillator_freqs.squeeze()[-100:].mean(axis=0),
            "vs": res_net.vs.squeeze(),
            "ws_final_var": np.var(
                oscillator_ranges_no_mod.squeeze()[-100:], axis=0),
            "fft_range": fft_range_detected[0],
            "convergence_time": convergence_time
        }
    
    trial_dict = {
        "experiment_config": config_dict,
        "radar_parameters": fmcw.get_parameters(),
        "network_parameters": res_net.get_parameters(),
        "targets": targets,
        "results": result_dict
    }
    stability = np.var(oscillator_ranges_no_mod.squeeze()[-100:], axis=0)
    if cfg.experiment.show_plot:

        fft_bins = fmcw.get_ranges()
        converged_threshold = (fft_bins[1] - fft_bins[0]) /2
        
        if len(targets)==1:
            error = np.abs(result_dict['range'] - targets[0]['range'])
            print(f"err: {error}")
        if len(targets)>1:
            avg_err = errors_n_targets(trial_dict)
            target_ranges = np.array([t['range'] for t in trial_dict['targets']])
            targets_stds = np.std(target_ranges)
            print(target_ranges)
            print(targets_stds)
            print(f"err: {fmcw.get_range_from_freq(avg_err)}, thresh: {converged_threshold}")
        print(f"snr: {fmcw.snr}")
        full_ws = ws_full.reshape((-1, res_net.n_units)) / (2 * np.pi)
        full_ws = np.mod(full_ws, fmcw.get_freq_from_range(fmcw.max_range))

        full_ranges = fmcw.get_range_from_freq(full_ws) 
        plt.plot(full_ranges, label='osc.', c="C0")
        
        for i in range(n_targets):
            plt.axhline(fft_range_detected[i], label='fft', c="C1")
        for targ in targets:
            plt.axhline(targ['range'], c="C2", label="gt")
        # plt.plot(mask)
        # plt.axvline(convergence_time)
        plt.axhline(max_range)
        # plt.axhline(0.0)
        # plt.legend()
        print(f"osc. time: {osc_time}")
        print(f"FFT time: {fft_time}")
        print(f"speedup: {fft_time / osc_time:.1f}")
        print(f"satbility: {np.var(oscillator_ranges_no_mod.squeeze()[-100:], axis=0)}")
        print(f"conv time: {result_dict['convergence_time']}")
        plt.show()

        
        # print(f"satbility: {result_dict['ws_final_var']}")
        # print(f"amplitude: {targets[0]['A_rand']}")

    return trial_dict

def init_worker():
    global rng
    rng = np.random.default_rng(
        int.from_bytes(os.urandom(4), "little")
    )
def run_trial(args):
    _, config_dict = args
    return run_sim(config_dict)

def main(config_path, save_path):
    config_dict = load_experiment_config(config_path)
    n_trials = config_dict["experiment"]["number_of_trials"]
    filename = config_dict["experiment"]["output_file"]
    save_file = config_dict['experiment']['save_file']

    if os.path.isfile(f"{save_path}/{filename}"):
        print(f"{filename} already exists, skipping.")
        return
    num_cores_to_use =  int(multiprocessing.cpu_count() * 0.8)
    with Pool(num_cores_to_use, initializer=init_worker) as pool:
        results = list(
            tqdm(
                pool.imap(
                    run_trial,
                    [(i, config_dict) for i in range(n_trials)],
                ),
                total=n_trials,
            )
        )

    if save_file:
        np.save(
            f"{save_path}/{filename}",
            results,
        )

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])