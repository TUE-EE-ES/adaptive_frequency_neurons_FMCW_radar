import numpy as np
from scipy import stats
from itertools import permutations

from src.fmcw_simulation.simulation import FmcwRadar

class ExperimentAnalyser:

    def __init__(self, file_path):

        self.file_path = file_path

        experiment_data = self.load_data()
        self.experiment_data = experiment_data
        self.radar_parameters = experiment_data[0]['radar_parameters']
        self.total_samples = self.radar_parameters['n_chirps'] * self.radar_parameters['n_samples']
        self.n_targets = len(experiment_data[0]['targets'])
        self.n_oscillators = experiment_data[0]['results']['range'].size
        self.experiment_config = experiment_data[0]['experiment_config']

        self.rmse = True

        self.extracted_data = self.extract_data(experiment_data)
        if self.n_oscillators == 1:
            self.extracted_data = self.extract_data_1target(experiment_data, self.extracted_data)
        if self.n_oscillators == 2:
            self.extracted_data = self.extract_data_2targets(experiment_data, self.extracted_data)
        if self.n_oscillators > 2:
            self.extracted_data = self.extract_data_n_targets(experiment_data, self.extracted_data)

    def measure_absolute_error(self, trial):
        gt_targets = trial['targets']
        results = trial['results']

        assert len(gt_targets) == 1
        fmcw = FmcwRadar(trial['experiment_config']['radar_config'])

        expected_frequency = fmcw.get_freq_from_range(gt_targets[0]['range'])
        measured_frequency = results['frequency']
        error = np.abs(expected_frequency - measured_frequency)
        
        return error

    def load_data(self):
        data_trials = np.load(self.file_path, allow_pickle=True)
        return data_trials
    
    def compute_bins(self, data_array, n_bins):

        bin_count, bin_edges, binnumber = stats.binned_statistic(self.data_array, self.data_array, 'count', bins=self.n_bins)

        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width/2

        return binnumber, bin_centers
    
    def group_by(self, group_name, n_bins=20):
        data_array = self.extracted_data[group_name]
        self.data_array = data_array
        self.n_bins = n_bins
        self.argbins, self.group_data_bins = self.compute_bins(data_array, n_bins)

        if self.n_oscillators == 1:
            self.errors_binned = self.apply_group_avg("final_error", self.argbins)
            self.rmse_binned = self.apply_group_rmse("final_error", self.argbins)
            self.std_binned = self.calculate_std_binned("final_error", self.argbins)

            self.fft_error_binned = self.apply_group_avg("fft_error", self.argbins)
            self.rmse_fft_binned = self.apply_group_rmse("fft_error", self.argbins)

            self.max_errors_binned = self.calculate_max_binned("final_error", self.argbins)
            self.frequency_variance_binned = self.apply_group_avg("frequency_variance", self.argbins)
            self.convergence_time_binned = self.apply_group_avg("convergence_time", self.argbins)
            self.convergence_speed_binned = self.apply_group_avg("convergence_speed", self.argbins)
            self.convergence_speed_binned_std = self.calculate_std_binned("convergence_speed", self.argbins)
            self.convergence_speed_binned_min = self.calculate_min_binned("convergence_speed", self.argbins)
            self.convergence_speed_binned_max = self.calculate_max_binned("convergence_speed", self.argbins)

        elif self.n_oscillators == 2:

            self.avg_errors_binned = self.apply_group_avg("avg_error", self.argbins)
            self.median_errors_binned = self.calculate_median_binned("avg_error", self.argbins)
            self.std_binned = self.calculate_std_binned("avg_error", self.argbins)
            self.max_errors_binned = self.calculate_max_binned("avg_error", self.argbins)
            self.rmse_binned = self.apply_group_rmse("avg_error", self.argbins)

            self.not_converged_binned = self.apply_group_avg("not_converged", self.argbins)
            self.not_converged_binned_var = self.calculate_std_binned("not_converged", self.argbins)
            self.not_converged_binned_median = self.calculate_median_binned("not_converged", self.argbins)
            
            self.frequency_variance_binned = self.apply_group_avg("frequency_variance", self.argbins)
            self.signal_amplitude_difference_binned = self.apply_group_avg("signal_amplitude_difference", self.argbins)
            self.signal_phase_difference_binned = self.apply_group_avg("signal_phase_difference", self.argbins)
            self.osc_start_freq_diff_binned = self.apply_group_avg("osc_start_freq_difference", self.argbins)

        elif self.n_oscillators > 2:
            self.avg_errors_binned = self.apply_group_avg("avg_error", self.argbins)
            self.median_errors_binned = self.calculate_median_binned("avg_error", self.argbins)
            self.std_binned = self.calculate_std_binned("avg_error", self.argbins)
            self.max_errors_binned = self.calculate_max_binned("avg_error", self.argbins)
            self.rmse_binned = self.apply_group_rmse("avg_error", self.argbins)
            
            self.frequency_variance_binned = self.apply_group_avg("frequency_variance", self.argbins)

    def apply_group_sum(self, group_name_apply, argbins):
        data_apply_array = self.extracted_data[group_name_apply]
        ret = stats.binned_statistic(self.data_array, data_apply_array, 'sum', bins=self.n_bins)
        return ret.statistic
    
    def apply_group_avg(self, group_name_apply, argbins):
        
        data_apply_array = self.extracted_data[group_name_apply]
        ret = stats.binned_statistic(self.data_array, data_apply_array, 'mean', bins=self.n_bins)
        return ret.statistic
    
    def apply_group_rmse(self, group_name_apply, argbins):
        
        data_apply_array = self.extracted_data[group_name_apply]
        data_squared = np.square(data_apply_array)
        ret = stats.binned_statistic(self.data_array, data_squared, 'mean', bins=self.n_bins)
        return np.sqrt(ret.statistic)
    
    def calculate_std_binned(self, group_name_apply, argbins):

        data_apply_array = self.extracted_data[group_name_apply]
        ret = stats.binned_statistic(self.data_array, data_apply_array, 'std', bins=self.n_bins)
        return ret.statistic
    
    def calculate_max_binned(self, group_name_apply, argbins):

        data_apply_array = self.extracted_data[group_name_apply]
        ret = stats.binned_statistic(self.data_array, data_apply_array, 'max', bins=self.n_bins)
        return ret.statistic
    
    def calculate_min_binned(self, group_name_apply, argbins):

        data_apply_array = self.extracted_data[group_name_apply]
        ret = stats.binned_statistic(self.data_array, data_apply_array, 'min', bins=self.n_bins)
        return ret.statistic
    
    def calculate_median_binned(self, group_name_apply, argbins):

        data_apply_array = self.extracted_data[group_name_apply]
        ret = stats.binned_statistic(self.data_array, data_apply_array, 'median', bins=self.n_bins)
        return ret.statistic
    
    def extract_data(self, data):
        
        if self.n_oscillators > 1:
            shape_data = (len(data))
        else:
            shape_data = (len(data))

        errors = np.zeros(shape_data)
        frequencies = np.zeros((len(data), self.n_oscillators))
        snr_levels = np.zeros(shape_data)
        varss = np.zeros(shape_data)
        fft_ranges_error = np.zeros(shape_data)
        fft_frequency_error = np.zeros(shape_data)
        nancounts = np.zeros(shape_data)
        gt_ranges = np.zeros(shape_data)
        w_scales = np.zeros(shape_data) 
        convergence_times = np.zeros(shape_data)

        for i, trial in enumerate(data):
            if self.n_targets == 1:
                osc_error = self.measure_absolute_error(trial)
            else:
                osc_error = 0
            if np.isnan(osc_error):
                nancounts[i] = 1
                continue
            fmcw = FmcwRadar(trial['experiment_config']['radar_config'])
            snr_levels[i] = trial['radar_parameters']['snr']
            errors[i] = osc_error
            frequencies[i] = trial['results']['frequency']
            varss[i] = np.mean(trial['results']['ws_final_var'])
            fft_ranges_error[i] = np.abs(trial['results']['fft_range'] - trial['targets'][0]['range'])
            fft_frequency_error[i] = np.abs(fmcw.get_freq_from_range(trial['results']['fft_range']) -
                                        fmcw.get_freq_from_range(trial['targets'][0]['range']))
            gt_ranges[i] = trial['targets'][0]['range']
            w_scale_norm = trial['network_parameters']['w_scale'].squeeze() # it should only be a single value for this test
            w_scale_norm /= trial['network_parameters']['t_res'] * 100
            w_scales[i] = np.mean(w_scale_norm)
            if trial['results']['convergence_time'] is not None:
                convergence_times[i] = trial['results']['convergence_time'] / self.total_samples
            else:
                convergence_times[i] = None
        return {
            "final_error": errors,
            "snr_level": snr_levels,
            "frequency_variance": varss.squeeze(),
            "fft_range_error": fft_ranges_error,
            "fft_error": fft_frequency_error,
            "nancount": nancounts,
            "gt_range": gt_ranges,
            "gt_freqs": fmcw.get_freq_from_range(gt_ranges),
            "w_scale": w_scales,
            "lambda": 1 / w_scales,
            "convergence_time": convergence_times,
            "final_frequencies": frequencies.flatten()
        }
    
    def extract_data_1target(self, data, extracted_data_dict):
        assert self.n_targets == 1

        starting_distance_to_target = np.zeros((len(data)))
        starting_range_osc = np.zeros((len(data)))
        signal_amplitude = np.zeros((len(data)))
        convergence_speed = np.zeros((len(data)))

        for i, trial in enumerate(data):
            starting_freq_hz = trial['network_parameters']['starting_frequencies'].squeeze()
            # we need to convert the freq in Hz to range to compare it with 
            # the ground truth range of the target
            fmcw = FmcwRadar(trial['radar_parameters'])
            starting_range_m = fmcw.get_range_from_freq(starting_freq_hz / (2 * np.pi))

            starting_distance_to_target[i] = np.abs(trial['targets'][0]['range'] - starting_range_m)
            starting_range_osc[i] = starting_range_m
            
            signal_amplitude[i] = trial['targets'][0]['A_rand']
            if trial['results']['convergence_time'] is not None:
                convergence_time = trial['results']['convergence_time'] #/ self.total_samples
                convergence_speed[i] = fmcw.get_freq_from_range(starting_distance_to_target[i]) / convergence_time
            else:
                convergence_speed[i] = None

        extracted_data_dict['starting_distance_to_target'] = starting_distance_to_target
        extracted_data_dict['starting_range_osc'] = starting_range_osc
        extracted_data_dict['signal_amplitude'] = signal_amplitude
        extracted_data_dict['convergence_speed'] = convergence_speed

        return extracted_data_dict

    
    def extract_data_2targets(self, data, extracted_data_dict):
        # some statistics are only relevant in the case there are 2 oscillaotrs/targets
        assert self.n_targets > 1 and self.n_oscillators > 1
        # distance between targets
        targets_distance = np.zeros((len(data)))
        average_error = np.zeros(len(data))
        not_converged = np.zeros(len(data))
        signal_amplitude_difference = np.zeros(len(data))
        signal_phase_difference = np.zeros(len(data))
        osc_start_freq_difference = np.zeros(len(data))
        targets_ratio = np.zeros((len(data)))
        starting_frequencies = np.zeros((len(data), 2))
        target_frequencies = np.zeros((len(data), 2))

        for i, trial in enumerate(data):
            targets_distance[i] = np.abs(trial['targets'][0]['range'] - trial['targets'][1]['range'])
            
            fmcw = FmcwRadar(trial['radar_parameters'])
            avg_err_trial, n_not_converged_trial = errors_2_targets(trial)
            average_error[i] = avg_err_trial
            not_converged[i] = n_not_converged_trial / self.n_targets
            signal_amplitude_difference[i] = np.abs(trial['targets'][0]['A_rand'] - trial['targets'][1]['A_rand'])
            signal_phase_difference[i] = np.abs(trial['targets'][0]['p_rand'] - trial['targets'][1]['p_rand'])
            osc_start_freq_difference[i] = np.abs(trial['network_parameters']['starting_frequencies'].squeeze()[0] - trial['network_parameters']['starting_frequencies'].squeeze()[1])
            starting_frequencies[i, 0] = trial['network_parameters']['starting_frequencies'].squeeze()[0]
            starting_frequencies[i, 1] = trial['network_parameters']['starting_frequencies'].squeeze()[1]
            ranges = np.array([trial['targets'][0]['range'], trial['targets'][1]['range']])
            target_frequencies[i] = fmcw.get_freq_from_range(ranges)
            targets_ratio[i] = ranges.max() / ranges.min()

        extracted_data_dict["targets_distance"] = targets_distance
        extracted_data_dict["avg_error"] = average_error
        extracted_data_dict["not_converged"] = not_converged
        # this is the difference in phase between the two frequencies in the input data
        extracted_data_dict['signal_amplitude_difference'] = signal_amplitude_difference
        extracted_data_dict['signal_phase_difference'] = signal_phase_difference
        extracted_data_dict['osc_start_freq_difference'] = osc_start_freq_difference
        extracted_data_dict['targets_ratio'] = targets_ratio
        extracted_data_dict['starting_frequencies'] = starting_frequencies
        extracted_data_dict['target_frequencies'] = target_frequencies

        return extracted_data_dict
    
    def extract_data_n_targets(self, data, extracted_data_dict):
        
        targets_distance = np.zeros((len(data)))
        average_error = np.zeros(len(data))
        n_targets = len(data[0]['targets'])
        starting_frequencies = np.zeros((len(data), n_targets))
        target_frequencies = np.zeros((len(data), n_targets))

        for i, trial in enumerate(data):
            targets_ranges = np.array([t['range'] for t in trial['targets']])
            min_distance = np.abs(targets_ranges[0] - targets_ranges[1])
            for r in targets_ranges:
                for r2 in targets_ranges:
                    if r == r2:
                        continue
                    dist = np.abs(r - r2)
                    if dist < min_distance:
                        min_distance = dist
            targets_distance[i] = min_distance
            
            fmcw = FmcwRadar(trial['radar_parameters'])
            avg_err_trial = errors_n_targets(trial)
            average_error[i] = avg_err_trial

            starting_frequencies[i] = trial['network_parameters']['starting_frequencies'].squeeze()
            target_frequencies[i] = fmcw.get_freq_from_range(targets_ranges)

        extracted_data_dict["targets_distance"] = targets_distance
        extracted_data_dict["avg_error"] = average_error
        extracted_data_dict['starting_frequencies'] = starting_frequencies
        extracted_data_dict['target_frequencies'] = target_frequencies

        return extracted_data_dict
def errors_n_targets(trial):
    gt_targets = trial['targets']
    results = trial['results']

    fmcw = FmcwRadar(trial['experiment_config']['radar_config'])

    targets_ranges = np.array([
        targ['range'] for targ in gt_targets
    ])
    expected_frequency = fmcw.get_freq_from_range(targets_ranges).squeeze()
    measured_frequency = results['frequency'].squeeze()

    # error = np.abs(expected_frequency.sum() - measured_frequency.sum()) #/ targets_ranges.size
    perms_measured_frequencies = np.array(list(permutations(measured_frequency)))
    errors_perms = np.abs(perms_measured_frequencies - expected_frequency).mean(axis=1)
    error = np.min(errors_perms)

    return error
        

def errors_2_targets(trial):
    
    return errors_n_targets(trial), 0.0
    # gt_targets = trial['targets']
    # results = trial['results']

    # assert len(gt_targets) == 2
    # fmcw = FmcwRadar(trial['experiment_config']['radar_config'])

    # targets_ranges = np.array([
    #     targ['range'] for targ in gt_targets
    # ])
    # expected_frequency = fmcw.get_freq_from_range(targets_ranges).squeeze()
    # measured_frequency = results['frequency'].squeeze()
    # error = np.abs(expected_frequency.sum() - measured_frequency.sum())
    # if error > 100000:
    #     print(f"expected: {expected_frequency}")
    #     print(f"measured: {measured_frequency}")
    # error1 = np.abs(expected_frequency - measured_frequency).sum() / 2
    # error2 = np.abs(expected_frequency - measured_frequency[::-1]).sum() / 2
    # error = min(error1, error2)
    
    # oscillator_ranges = oscillators_dict['range']
    # targets_ranges = np.array([
    #     targ['range'] for targ in targets_dict
    # ])
    # errors = []
    # number_not_converged = 0

    # oscillator_frequency = 

    # for targ_range in targets_ranges:
    #     ranges_diff = np.abs(oscillator_ranges - targ_range)
    #     sorted_idxs = np.argsort(ranges_diff)
    #     sorted_diff = ranges_diff[sorted_idxs]

    #     errors.append(sorted_diff[0])
    #     oscillator_ranges = np.delete(oscillator_ranges, sorted_idxs[0])

    #     if not np.isnan(sorted_diff[0]) and sorted_diff[0] < threshold:
    #         test = 1
    #     else:
    #         number_not_converged += 1
    
    return error, 0.0