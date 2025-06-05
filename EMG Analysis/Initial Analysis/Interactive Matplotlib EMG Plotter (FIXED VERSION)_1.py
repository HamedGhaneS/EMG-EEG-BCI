#!/usr/bin/env python3
"""
Interactive Matplotlib EMG Plotter - COMPLETE UPDATED VERSION

This script creates interactive matplotlib plots with automatic CSV export.
TWO-PHASE APPROACH:
1. Analyzes ALL trials and saves complete CSV first
2. Optional interactive plot viewing

Requirements:
- mne
- numpy
- matplotlib
- scipy
- pandas

Features:
- Complete CSV export with all trials (guaranteed)
- Interactive zooming/panning with matplotlib toolbar
- Two panels: Raw EMG + RMS envelope
- Robust EMG burst detection with conservative thresholds
- Smart EMG onset detection closest to button press
- All threshold crossings recorded for manual correction
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from scipy import signal as scipy_signal
import pandas as pd
import csv
warnings.filterwarnings('ignore')

# Use interactive backend
plt.ion()  # Turn on interactive mode

class InteractiveEMGPlotter:
    def __init__(self, eeg_data_path):
        """
        Initialize interactive EMG plotter
        
        Parameters:
        eeg_data_path (str): Path to BrainVision EEG files
        """
        self.eeg_data_path = Path(eeg_data_path)
        
        # Trigger mapping
        self.trigger_mapping = {
            10: 'trial_start', 20: 'coherent_motion_start', 101: 'button_press_left',
            102: 'button_press_right', 111: 'confidence_low', 112: 'confidence_med',
            113: 'confidence_high', 200: 'trial_end', 250: 'block_start',
            251: 'block_end', 254: 'experiment_start', 255: 'experiment_end'
        }
        
        # Colors for triggers
        self.trigger_colors = {
            10: '#0000FF', 20: '#FF8000', 101: '#FF00FF', 102: '#800080',
            111: '#FFFF00', 112: '#FFD700', 113: '#FFA500', 200: '#000080',
            250: '#00FF00', 251: '#008000', 254: '#FF0000', 255: '#800000'
        }
    
    def load_data(self, eeg_filename):
        """Load EEG data and extract events"""
        vhdr_file = self.eeg_data_path / f"{eeg_filename}.vhdr"
        raw = mne.io.read_raw_brainvision(vhdr_file, preload=True, verbose=False)
        
        # Extract events
        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        mapped_events = self.map_event_codes(events, event_dict)
        
        # Extract EMG
        emg_channels = [ch for ch in raw.ch_names if 'EMG' in ch.upper()]
        emg_data = raw.get_data(picks=[emg_channels[0]])[0]
        times = raw.times
        
        print(f"Loaded {eeg_filename}: {len(times)/raw.info['sfreq']:.1f}s, {len(events)} events")
        
        return raw, emg_data, times, mapped_events, emg_channels[0]
    
    def map_event_codes(self, events, event_dict):
        """Map MNE event codes to original trigger codes"""
        code_mapping = {}
        
        for event_name, mne_code in event_dict.items():
            if 'S' in event_name:
                try:
                    s_part = event_name.split('S')[-1].strip()
                    original_code = int(s_part)
                    code_mapping[mne_code] = original_code
                except ValueError:
                    if 'New Segment' in event_name:
                        code_mapping[mne_code] = 99999
        
        # Map events
        mapped_events = events.copy()
        for i, (sample, prev, mne_code) in enumerate(events):
            if mne_code in code_mapping:
                mapped_events[i, 2] = code_mapping[mne_code]
        
        return mapped_events
    
    def filter_short_bursts(self, burst_signal, min_samples):
        """Remove bursts shorter than min_samples"""
        filtered_signal = burst_signal.copy()
        
        # Find burst start and end points
        diff_signal = np.diff(burst_signal.astype(int))
        burst_starts = np.where(diff_signal == 1)[0] + 1
        burst_ends = np.where(diff_signal == -1)[0] + 1
        
        # Handle edge cases
        if len(burst_starts) == 0:
            return filtered_signal
        
        if burst_signal[0]:  # Signal starts with a burst
            burst_starts = np.insert(burst_starts, 0, 0)
        
        if burst_signal[-1]:  # Signal ends with a burst
            burst_ends = np.append(burst_ends, len(burst_signal))
        
        # Ensure equal number of starts and ends
        min_len = min(len(burst_starts), len(burst_ends))
        burst_starts = burst_starts[:min_len]
        burst_ends = burst_ends[:min_len]
        
        # Remove short bursts
        for start, end in zip(burst_starts, burst_ends):
            if end - start < min_samples:
                filtered_signal[start:end] = False
        
        return filtered_signal
    
    def find_burst_periods_simple(self, times, burst_signal):
        """Find continuous burst periods"""
        burst_periods = []
        in_burst = False
        burst_start = None
        
        for i, is_burst in enumerate(burst_signal):
            if is_burst and not in_burst:
                burst_start = times[i]
                in_burst = True
            elif not is_burst and in_burst:
                if burst_start is not None:
                    burst_periods.append((burst_start, times[i-1]))
                in_burst = False
        
        # Handle case where burst continues to end
        if in_burst and burst_start is not None:
            burst_periods.append((burst_start, times[-1]))
        
        return burst_periods
    
    def find_all_threshold_crossings(self, times, rms_signal, threshold):
        """
        Find all threshold crossings (upward transitions) in the RMS signal
        
        Returns:
        crossing_times: List of times when signal crosses threshold upward
        """
        crossing_times = []
        
        # Find upward threshold crossings
        below_threshold = rms_signal < threshold
        above_threshold = rms_signal >= threshold
        
        # Find transitions from below to above threshold
        transitions = np.diff(below_threshold.astype(int))
        crossing_indices = np.where(transitions == -1)[0] + 1  # +1 because diff shifts indices
        
        # Convert to times
        crossing_times = [times[idx] for idx in crossing_indices if idx < len(times)]
        
        return crossing_times
    
    def save_results_to_csv(self, all_results, participant_name, output_dir=None):
        """
        Save all trial results to CSV file
        
        Parameters:
        all_results: List of dictionaries containing trial results
        participant_name: Name of participant for filename
        output_dir: Directory to save CSV (default: same as script)
        """
        if output_dir is None:
            output_dir = Path.cwd()
        else:
            output_dir = Path(output_dir)
        
        # Create output filename
        csv_filename = output_dir / f"{participant_name}_EMG_Analysis_Results.csv"
        
        # Define CSV columns
        columns = [
            'Participant', 'Block', 'Trial', 
            'Trial_Start_Time', 'Motion_Start_Time', 'Button_Press_Time', 'Trial_End_Time',
            'EMG_Onset_Time', 'EMG_to_Button_Delay_ms',
            'Threshold_Crossings_Count', 'All_Threshold_Crossings_Times',
            'Burst_Periods_Count', 'All_Burst_Periods',
            'Baseline_Median', 'Baseline_MAD', 'Final_Threshold',
            'Motion_to_Button_RT_ms', 'Trial_to_Motion_Delay_ms', 'EMG_after_Motion_ms'
        ]
        
        # Prepare data for CSV
        csv_data = []
        
        for result in all_results:
            # Calculate additional timing metrics
            motion_to_button_rt = None
            trial_to_motion_delay = None
            emg_after_motion = None
            
            if result['Motion_Start_Time'] and result['Button_Press_Time']:
                motion_to_button_rt = (result['Button_Press_Time'] - result['Motion_Start_Time']) * 1000
            
            if result['Trial_Start_Time'] and result['Motion_Start_Time']:
                trial_to_motion_delay = (result['Motion_Start_Time'] - result['Trial_Start_Time']) * 1000
            
            if result['EMG_Onset_Time'] and result['Motion_Start_Time']:
                emg_after_motion = (result['EMG_Onset_Time'] - result['Motion_Start_Time']) * 1000
            
            # Format threshold crossings and burst periods as strings
            crossings_str = ';'.join([f"{t:.3f}" for t in result.get('All_Threshold_Crossings_Times', [])])
            burst_periods_str = ';'.join([f"{start:.3f}-{end:.3f}" for start, end in result.get('All_Burst_Periods', [])])
            
            row = {
                'Participant': result['Participant'],
                'Block': result['Block'],
                'Trial': result['Trial'],
                'Trial_Start_Time': f"{result['Trial_Start_Time']:.3f}" if result['Trial_Start_Time'] else "",
                'Motion_Start_Time': f"{result['Motion_Start_Time']:.3f}" if result['Motion_Start_Time'] else "",
                'Button_Press_Time': f"{result['Button_Press_Time']:.3f}" if result['Button_Press_Time'] else "",
                'Trial_End_Time': f"{result['Trial_End_Time']:.3f}" if result['Trial_End_Time'] else "",
                'EMG_Onset_Time': f"{result['EMG_Onset_Time']:.3f}" if result['EMG_Onset_Time'] else "",
                'EMG_to_Button_Delay_ms': f"{result['EMG_to_Button_Delay_ms']:.0f}" if result['EMG_to_Button_Delay_ms'] else "",
                'Threshold_Crossings_Count': result.get('Threshold_Crossings_Count', 0),
                'All_Threshold_Crossings_Times': crossings_str,
                'Burst_Periods_Count': result.get('Burst_Periods_Count', 0),
                'All_Burst_Periods': burst_periods_str,
                'Baseline_Median': f"{result.get('Baseline_Median', 0):.6f}",
                'Baseline_MAD': f"{result.get('Baseline_MAD', 0):.6f}",
                'Final_Threshold': f"{result.get('Final_Threshold', 0):.6f}",
                'Motion_to_Button_RT_ms': f"{motion_to_button_rt:.0f}" if motion_to_button_rt else "",
                'Trial_to_Motion_Delay_ms': f"{trial_to_motion_delay:.0f}" if trial_to_motion_delay else "",
                'EMG_after_Motion_ms': f"{emg_after_motion:.0f}" if emg_after_motion else ""
            }
            
            csv_data.append(row)
        
        # Write to CSV
        try:
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()
                writer.writerows(csv_data)
            
            print(f"✅ Results saved to: {csv_filename}")
            print(f"📊 Total trials exported: {len(csv_data)}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving CSV: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def analyze_trial_for_csv(self, emg_data, times, trial_info, block_num, participant_name, sfreq):
        """
        Analyze a single trial and return results for CSV export
        """
        # Detect EMG bursts
        result = self.detect_emg_bursts_trial_specific(emg_data, times, trial_info, sfreq)
        
        if len(result) == 7:  # New format with onset detection
            burst_signal, rms_signal, threshold, baseline_stats, emg_filtered, emg_onset_time, onset_to_button_delay = result
        else:  # Fallback format
            burst_signal, rms_signal, threshold, baseline_stats, emg_filtered = result[:5]
            emg_onset_time, onset_to_button_delay = None, None
        
        # Find all threshold crossings
        threshold_crossings = self.find_all_threshold_crossings(times, rms_signal, threshold)
        
        # Find all burst periods
        burst_periods = self.find_burst_periods_simple(times, burst_signal)
        
        # Prepare result dictionary
        trial_result = {
            'Participant': participant_name,
            'Block': block_num,
            'Trial': trial_info['trial_num'],
            'Trial_Start_Time': trial_info.get('trial_start'),
            'Motion_Start_Time': trial_info.get('motion_start'),
            'Button_Press_Time': trial_info.get('button_press'),
            'Trial_End_Time': trial_info.get('trial_end'),
            'EMG_Onset_Time': emg_onset_time,
            'EMG_to_Button_Delay_ms': onset_to_button_delay * 1000 if onset_to_button_delay else None,
            'Threshold_Crossings_Count': len(threshold_crossings),
            'All_Threshold_Crossings_Times': threshold_crossings,
            'Burst_Periods_Count': len(burst_periods),
            'All_Burst_Periods': burst_periods,
            'Baseline_Median': baseline_stats.get('median'),
            'Baseline_MAD': baseline_stats.get('mad'),
            'Final_Threshold': baseline_stats.get('threshold')
        }
        
        return trial_result
    
    def detect_emg_bursts_trial_specific(self, emg_data, times, trial_info, sfreq):
        """
        Improved EMG burst detection with high-pass filtering and trial-specific thresholds
        Fixed version incorporating all debugging improvements
        
        Parameters:
        emg_data: Raw EMG signal
        times: Time array
        trial_info: Dictionary with trial timing information
        sfreq: Sampling frequency
        
        Returns:
        burst_signal: Binary array indicating burst periods
        rms_signal: RMS envelope
        threshold: Burst detection threshold
        baseline_stats: Dictionary with baseline statistics
        emg_filtered: High-pass filtered EMG signal
        emg_onset_time: Time of EMG burst onset (or None)
        onset_to_button_delay: Time from EMG onset to button press (or None)
        """
        
        # Step 1: High-pass filter to remove low-frequency fluctuations
        nyquist = sfreq / 2
        high_cutoff = 10.0  # Hz - removes DC drift and slow fluctuations
        
        # Use 4th order Butterworth high-pass filter
        sos = scipy_signal.butter(4, high_cutoff/nyquist, btype='high', output='sos')
        emg_filtered = scipy_signal.sosfiltfilt(sos, emg_data)
        
        # Step 2: Calculate RMS envelope from filtered signal
        window_samples = int(0.02 * sfreq)  # 20ms window
        emg_squared = emg_filtered ** 2
        rms_signal = np.sqrt(np.convolve(emg_squared, np.ones(window_samples)/window_samples, mode='same'))
        
        # Step 3: Trial-specific threshold calculation
        # Define baseline period for this specific trial (before motion start)
        if trial_info['motion_start'] and trial_info['trial_start']:
            # Use period from trial start to motion start as baseline
            baseline_start_time = trial_info['trial_start']
            baseline_end_time = trial_info['motion_start'] - 0.1  # 100ms before motion
            
            # Ensure minimum baseline duration of 1 second
            if baseline_end_time - baseline_start_time < 1.0:
                baseline_start_time = baseline_end_time - 1.0
        else:
            # Fallback: use first part of the trial window
            trial_start_time = trial_info.get('trial_start', times[0])
            baseline_start_time = trial_start_time - 1.0
            baseline_end_time = trial_start_time + 0.5
        
        # Convert to sample indices
        baseline_start_idx = int(max(0, (baseline_start_time - times[0]) * sfreq))
        baseline_end_idx = int(min(len(rms_signal), (baseline_end_time - times[0]) * sfreq))
        
        # Ensure indices are within bounds
        if baseline_end_idx <= baseline_start_idx:
            baseline_start_idx = int(1.0 * sfreq)
            baseline_end_idx = min(int(3.0 * sfreq), len(rms_signal))
        
        # Step 4: Robust baseline statistics
        baseline_rms = rms_signal[baseline_start_idx:baseline_end_idx]
        
        if len(baseline_rms) == 0:
            return self.detect_emg_bursts_global_fallback(emg_data, times, sfreq)
        
        # Use robust statistics
        baseline_median = np.median(baseline_rms)
        baseline_mad = np.median(np.abs(baseline_rms - baseline_median))
        baseline_95th = np.percentile(baseline_rms, 95)
        baseline_99th = np.percentile(baseline_rms, 99)
        
        # Step 5: Conservative threshold calculation (FIXED)
        # Use more conservative threshold to avoid false positives
        threshold_mad = baseline_median + 6 * baseline_mad  # More conservative (was 4)
        threshold_percentile = baseline_95th * 2.0  # Higher multiplier (was 1.5)
        
        # Select final threshold (most conservative)
        threshold = max(threshold_mad, threshold_percentile)
        
        # Ensure minimum threshold to avoid noise
        min_threshold = baseline_median + 3 * baseline_mad
        threshold = max(threshold, min_threshold)
        
        # Step 6: Burst detection with minimum duration requirement (FIXED)
        burst_signal_raw = rms_signal > threshold
        
        # Apply minimum burst duration (50ms) to filter out brief spikes
        min_burst_samples = int(0.05 * sfreq)  # 50ms minimum
        burst_signal = self.filter_short_bursts(burst_signal_raw, min_burst_samples)
        
        n_burst_samples = np.sum(burst_signal)
        
        # Step 7: Smart EMG onset detection (FIXED)
        emg_onset_time = None
        onset_to_button_delay = None
        
        if trial_info.get('button_press') and n_burst_samples > 0:
            button_time = trial_info['button_press']
            
            # Find all burst periods
            burst_periods = self.find_burst_periods_simple(times, burst_signal)
            
            if burst_periods:
                # Find burst closest to (but before) button press
                valid_bursts = [(start, end) for start, end in burst_periods if start < button_time]
                
                if valid_bursts:
                    # Get the burst closest to button press
                    closest_burst = max(valid_bursts, key=lambda x: x[0])  # Latest start time
                    emg_onset_time = closest_burst[0]
                    onset_to_button_delay = button_time - emg_onset_time
        
        baseline_stats = {
            'median': baseline_median,
            'mad': baseline_mad,
            'threshold': threshold,
            'lower_threshold': threshold * 0.6,
            'baseline_95th': baseline_95th,
            'baseline_99th': baseline_99th,
            'baseline_start_time': baseline_start_time,
            'baseline_end_time': baseline_end_time
        }
        
        return burst_signal, rms_signal, threshold, baseline_stats, emg_filtered, emg_onset_time, onset_to_button_delay
    
    def detect_emg_bursts_global_fallback(self, emg_data, times, sfreq):
        """Fallback to global threshold if trial-specific fails"""
        # Simple global approach
        window_samples = int(0.02 * sfreq)
        emg_squared = emg_data ** 2
        rms_signal = np.sqrt(np.convolve(emg_squared, np.ones(window_samples)/window_samples, mode='same'))
        
        baseline_start = int(1.0 * sfreq)
        baseline_end = int(5.0 * sfreq)
        baseline_rms = rms_signal[baseline_start:baseline_end]
        
        baseline_median = np.median(baseline_rms)
        baseline_mad = np.median(np.abs(baseline_rms - baseline_median))
        threshold = baseline_median + 6 * baseline_mad  # More conservative
        
        burst_signal = rms_signal > threshold
        
        baseline_stats = {
            'median': baseline_median,
            'mad': baseline_mad,
            'threshold': threshold,
            'lower_threshold': threshold * 0.7
        }
        
        return burst_signal, rms_signal, threshold, baseline_stats, emg_data, None, None
    
    def find_trials_by_blocks(self, events):
        """
        Find trials organized by blocks
        
        Returns:
        blocks_data: Dictionary with block information and trials
        """
        sfreq = 5000.0  # Sampling frequency
        trigger_times = events[:, 0] / sfreq
        trigger_codes = events[:, 2]
        
        # Find block starts
        block_starts = trigger_times[trigger_codes == 250]
        
        # Find all trial events
        trial_starts = trigger_times[trigger_codes == 10]
        motion_starts = trigger_times[trigger_codes == 20]
        button_presses = trigger_times[trigger_codes == 101]
        trial_ends = trigger_times[trigger_codes == 200]
        
        blocks_data = {}
        
        for block_idx, block_start in enumerate(block_starts):
            block_num = block_idx + 1
            
            # Find trials in this block
            if block_idx < len(block_starts) - 1:
                next_block_start = block_starts[block_idx + 1]
                block_trials = trial_starts[(trial_starts >= block_start) & (trial_starts < next_block_start)]
            else:
                block_trials = trial_starts[trial_starts >= block_start]
            
            trials_info = []
            
            for trial_idx, trial_start in enumerate(block_trials):
                trial_num = trial_idx + 1
                
                # Find corresponding events for this trial
                motion_start = None
                button_press = None
                trial_end = None
                
                # Find next motion start after trial start
                motion_candidates = motion_starts[motion_starts > trial_start]
                if len(motion_candidates) > 0:
                    motion_start = motion_candidates[0]
                
                # Find next button press after motion start
                if motion_start:
                    button_candidates = button_presses[button_presses > motion_start]
                    if len(button_candidates) > 0:
                        button_press = button_candidates[0]
                
                # Find next trial end
                end_candidates = trial_ends[trial_ends > trial_start]
                if len(end_candidates) > 0:
                    trial_end = end_candidates[0]
                
                trials_info.append({
                    'trial_num': trial_num,
                    'trial_start': trial_start,
                    'motion_start': motion_start,
                    'button_press': button_press,
                    'trial_end': trial_end,
                })
            
            blocks_data[block_num] = {
                'block_start': block_start,
                'trials': trials_info
            }
        
        return blocks_data
    
    def add_trigger_lines(self, ax, trigger_times, trigger_codes, legend_loc='upper right'):
        """Add trigger lines to plot"""
        unique_codes = np.unique(trigger_codes)
        trigger_legend_added = set()
        
        for code in unique_codes:
            if code == 99999:  # Skip new segment
                continue
            
            code_mask = trigger_codes == code
            code_times = trigger_times[code_mask]
            
            color = self.trigger_colors.get(code, '#000000')
            trigger_name = self.trigger_mapping.get(code, f'Code_{code}')
            
            for i, t in enumerate(code_times):
                label = trigger_name if code not in trigger_legend_added else ""
                ax.axvline(x=t, color=color, linestyle='--', alpha=0.8, linewidth=2, label=label)
                
                if code not in trigger_legend_added:
                    trigger_legend_added.add(code)
        
        if len(trigger_legend_added) > 0:
            ax.legend(loc=legend_loc, fontsize=10)
    
    def add_trial_markers(self, ax, trial_info):
        """Add trial event markers"""
        if trial_info['trial_start']:
            ax.axvline(x=trial_info['trial_start'], color='blue', linestyle=':', linewidth=3, 
                      alpha=0.8, label='Trial Start')
        
        if trial_info['motion_start']:
            ax.axvline(x=trial_info['motion_start'], color='orange', linestyle=':', linewidth=3,
                      alpha=0.8, label='Motion Start')
        
        if trial_info['button_press']:
            ax.axvline(x=trial_info['button_press'], color='magenta', linestyle=':', linewidth=3,
                      alpha=0.8, label='Button Press')
        
        if trial_info['trial_end']:
            ax.axvline(x=trial_info['trial_end'], color='darkblue', linestyle=':', linewidth=3,
                      alpha=0.8, label='Trial End')
    
    def plot_trial_interactive(self, emg_data, times, events, trial_info, block_num, run_name):
        """
        Create interactive plot for a single trial (FIXED VERSION)
        """
        sfreq = len(times) / (times[-1] - times[0])
        trigger_times = events[:, 0] / sfreq
        trigger_codes = events[:, 2]
        
        trial = trial_info
        trial_num = trial['trial_num']
        
        print(f"\n📊 Plotting Block {block_num}, Trial {trial_num}...")
        
        # Detect EMG bursts using fixed method
        result = self.detect_emg_bursts_trial_specific(emg_data, times, trial, sfreq)
        
        if len(result) == 7:  # New format with onset detection
            burst_signal, rms_signal, threshold, baseline_stats, emg_filtered, emg_onset_time, onset_to_button_delay = result
        else:  # Fallback format
            burst_signal, rms_signal, threshold, baseline_stats, emg_filtered = result[:5]
            emg_onset_time, onset_to_button_delay = None, None
        
        # Define time window (motion start to 2 seconds after button press)
        if trial['motion_start'] and trial['button_press']:
            window_start = trial['motion_start'] - 1.0
            window_end = trial['button_press'] + 2.0
        elif trial['trial_start']:
            window_start = trial['trial_start'] - 0.5
            window_end = trial['trial_start'] + 5.0
        else:
            print(f"⚠️  No valid timing for Block {block_num}, Trial {trial_num}")
            return None
        
        # Extract data for this window
        time_mask = (times >= window_start) & (times <= window_end)
        times_trial = times[time_mask]
        emg_trial = emg_data[time_mask]
        burst_trial = burst_signal[time_mask]
        rms_trial = rms_signal[time_mask]
        
        # Extract triggers for this window
        trigger_mask = (trigger_times >= window_start) & (trigger_times <= window_end)
        trigger_times_trial = trigger_times[trigger_mask]
        trigger_codes_trial = trigger_codes[trigger_mask]
        
        # Create interactive figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f'{run_name} - Block {block_num}, Trial {trial_num} (Interactive - Use Toolbar to Zoom)', 
                    fontsize=14, fontweight='bold')
        
        # Plot 1: Raw EMG (filtered if available) with burst highlighting
        if emg_filtered is not None:
            ax1.plot(times_trial, emg_trial, 'lightblue', linewidth=0.5, alpha=0.6, label='Raw EMG')
            # Use filtered EMG for display
            emg_filt_trial = emg_filtered[time_mask]
            ax1.plot(times_trial, emg_filt_trial, 'b-', linewidth=1, alpha=0.8, label='Filtered EMG (>10Hz)')
        else:
            ax1.plot(times_trial, emg_trial, 'b-', linewidth=1, alpha=0.8, label='Raw EMG')
        
        # Highlight burst periods (FIXED)
        burst_periods = self.find_burst_periods_simple(times_trial, burst_trial)
        if burst_periods:
            for i, (start_time, end_time) in enumerate(burst_periods):
                ax1.axvspan(start_time, end_time, alpha=0.3, color='yellow',
                           label='EMG Burst' if i == 0 else "")
        else:
            # Add a note if no bursts detected
            ax1.text(0.02, 0.98, 'No EMG bursts detected', transform=ax1.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7),
                    verticalalignment='top', fontsize=10)
        
        # Add EMG burst onset marker (THINNER LINE)
        if emg_onset_time and window_start <= emg_onset_time <= window_end:
            ax1.axvline(x=emg_onset_time, color='red', linestyle='-', linewidth=1.5,
                       alpha=0.9, label=f'EMG Onset ({emg_onset_time:.3f}s)')
            
            # Add onset to RMS plot too
            ax2.axvline(x=emg_onset_time, color='red', linestyle='-', linewidth=1.5,
                       alpha=0.9, label=f'EMG Onset')
        
        # Add triggers and trial markers
        self.add_trigger_lines(ax1, trigger_times_trial, trigger_codes_trial, 'upper left')
        self.add_trial_markers(ax1, trial)
        
        ax1.set_ylabel('EMG Amplitude (V)', fontsize=12)
        ax1.set_title('Raw EMG Signal with Burst Detection', fontsize=13)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: RMS envelope with multiple thresholds
        ax2.plot(times_trial, rms_trial, 'g-', linewidth=2, label='RMS Envelope')
        ax2.axhline(y=baseline_stats['threshold'], color='r', linestyle='--', linewidth=2,
                   alpha=0.8, label=f'Final Threshold ({baseline_stats["threshold"]:.6f})')
        ax2.axhline(y=baseline_stats['lower_threshold'], color='orange', linestyle=':', linewidth=2,
                   alpha=0.6, label=f'Lower Threshold ({baseline_stats["lower_threshold"]:.6f})')
        
        # Add baseline period highlighting if available
        if 'baseline_start_time' in baseline_stats and 'baseline_end_time' in baseline_stats:
            baseline_start = baseline_stats['baseline_start_time'] 
            baseline_end = baseline_stats['baseline_end_time']
            if baseline_start >= window_start and baseline_end <= window_end:
                ax2.axvspan(baseline_start, baseline_end, alpha=0.2, color='cyan', 
                           label=f'Baseline Period')
        
        # Add triggers and trial markers
        self.add_trigger_lines(ax2, trigger_times_trial, trigger_codes_trial, 'upper left')
        self.add_trial_markers(ax2, trial)
        
        ax2.set_ylabel('RMS Amplitude (V)', fontsize=12)
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_title('RMS Envelope with Burst Threshold', fontsize=13)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Set same x-axis limits for both plots
        for ax in [ax1, ax2]:
            ax.set_xlim(window_start, window_end)
        
        # Add timing information with EMG onset details
        timing_text = self.create_timing_info_text(trial, emg_onset_time, onset_to_button_delay)
        fig.text(0.02, 0.02, timing_text, fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for timing info
        
        # Show the plot
        plt.show()
        
        return fig
    
    def create_timing_info_text(self, trial, emg_onset_time=None, onset_to_button_delay=None):
        """Create timing information text with EMG onset details"""
        text_lines = [
            f"Trial {trial['trial_num']} - Timing Information:",
        ]
        
        if trial['trial_start']:
            text_lines.append(f"Trial Start: {trial['trial_start']:.3f}s")
        
        if trial['motion_start'] and trial['trial_start']:
            delay = trial['motion_start'] - trial['trial_start']
            text_lines.append(f"Motion Start: {trial['motion_start']:.3f}s (Δ{delay:.3f}s)")
        
        if emg_onset_time:
            text_lines.append(f"EMG Onset: {emg_onset_time:.3f}s")
            
            if trial['motion_start']:
                emg_after_motion = emg_onset_time - trial['motion_start']
                text_lines.append(f"EMG after Motion: {emg_after_motion:.3f}s ({emg_after_motion*1000:.0f}ms)")
        
        if trial['button_press']:
            if trial['motion_start']:
                rt = trial['button_press'] - trial['motion_start']
                text_lines.append(f"Button Press: {trial['button_press']:.3f}s (RT: {rt:.3f}s)")
            
            if onset_to_button_delay:
                text_lines.append(f"EMG-to-Button: {onset_to_button_delay:.3f}s ({onset_to_button_delay*1000:.0f}ms)")
        
        return "  |  ".join(text_lines)
    
    def run_interactive_analysis(self, run_configs):
        """Run interactive analysis for each participant - TWO-PHASE APPROACH"""
        print("INTERACTIVE EMG ANALYSIS - COMPLETE UPDATED VERSION")
        print("=" * 60)
        print("🔍 Interactive matplotlib plots with zoom/pan capability")
        print("📊 Two panels: Raw EMG + RMS envelope")
        print("🎯 Use matplotlib toolbar to zoom and pan")
        print("⌨️  Press Enter to continue to next trial, 'q' to quit, 's' to skip block")
        print("🔧 All debugging fixes applied for robust detection")
        print("💾 ALL trials analyzed and CSV saved BEFORE plotting begins")
        print("=" * 60)
        
        for config in run_configs:
            print(f"\n🚀 Starting analysis for {config['name']}...")
            
            try:
                # Load data
                raw, emg_data, times, events, emg_channel = self.load_data(config['eeg_file'])
                
                # Find trials organized by blocks
                blocks_data = self.find_trials_by_blocks(events)
                
                print(f"📈 Found {len(blocks_data)} blocks")
                print(f"🎵 Using EMG channel: {emg_channel}")
                
                # STEP 1: ANALYZE ALL TRIALS AND SAVE CSV FIRST
                print(f"\n🔬 ANALYZING ALL TRIALS FOR CSV EXPORT...")
                participant_results = []
                total_trials = sum(len(block_info['trials']) for block_info in blocks_data.values())
                
                trial_count = 0
                for block_num, block_info in blocks_data.items():
                    trials = block_info['trials']
                    print(f"  📋 Analyzing Block {block_num}: {len(trials)} trials...")
                    
                    for trial_info in trials:
                        trial_count += 1
                        trial_num = trial_info['trial_num']
                        
                        try:
                            # Analyze trial for CSV (this runs the detection)
                            print(f"    🔧 Block {block_num}, Trial {trial_num} ({trial_count}/{total_trials})")
                            trial_result = self.analyze_trial_for_csv(emg_data, times, trial_info, 
                                                                    block_num, config['name'], raw.info['sfreq'])
                            participant_results.append(trial_result)
                            
                        except Exception as e:
                            print(f"    ❌ Error analyzing Block {block_num}, Trial {trial_num}: {e}")
                            # Still add a placeholder entry
                            placeholder_result = {
                                'Participant': config['name'],
                                'Block': block_num,
                                'Trial': trial_info['trial_num'],
                                'Trial_Start_Time': trial_info.get('trial_start'),
                                'Motion_Start_Time': trial_info.get('motion_start'),
                                'Button_Press_Time': trial_info.get('button_press'),
                                'Trial_End_Time': trial_info.get('trial_end'),
                                'EMG_Onset_Time': None,
                                'EMG_to_Button_Delay_ms': None,
                                'Threshold_Crossings_Count': 0,
                                'All_Threshold_Crossings_Times': [],
                                'Burst_Periods_Count': 0,
                                'All_Burst_Periods': [],
                                'Baseline_Median': None,
                                'Baseline_MAD': None,
                                'Final_Threshold': None
                            }
                            participant_results.append(placeholder_result)
                
                # SAVE CSV IMMEDIATELY AFTER ALL ANALYSIS IS COMPLETE
                print(f"\n💾 SAVING CSV WITH ALL {len(participant_results)} TRIALS...")
                csv_saved = self.save_results_to_csv(participant_results, config['name'], self.eeg_data_path)
                
                if csv_saved:
                    print(f"✅ CSV SUCCESSFULLY SAVED! You can now browse plots or quit anytime.")
                else:
                    print(f"❌ CSV SAVE FAILED! Check the error messages above.")
                
                # STEP 2: INTERACTIVE PLOTTING (OPTIONAL)
                print(f"\n🖼️  STARTING INTERACTIVE PLOT VIEWING...")
                print(f"📝 Note: CSV is already saved. You can quit anytime without losing data.")
                
                # Ask user if they want to view plots
                view_plots = input(f"\nDo you want to view interactive plots? [y/n] (default: y): ").strip().lower()
                if view_plots in ['n', 'no']:
                    print("⏭️  Skipping plot viewing. CSV data is already saved!")
                    continue
                
                # Process each block for plotting
                for block_num, block_info in blocks_data.items():
                    trials = block_info['trials']
                    print(f"\n📋 Block {block_num}: {len(trials)} trials")
                    
                    for trial_info in trials:
                        trial_num = trial_info['trial_num']
                        
                        try:
                            # Create interactive plot
                            fig = self.plot_trial_interactive(emg_data, times, events, trial_info, 
                                                            block_num, config['name'])
                            
                            if fig:
                                # Wait for user input
                                user_input = input(f"\n⏳ Block {block_num}, Trial {trial_num} displayed. [Enter]=next, 'q'=quit, 's'=skip block: ").strip().lower()
                                
                                plt.close(fig)  # Close current figure
                                
                                if user_input == 'q':
                                    print("👋 Quitting plot viewing...")
                                    print("💾 CSV already saved - no data lost!")
                                    return
                                elif user_input == 's':
                                    print(f"⏭️  Skipping rest of Block {block_num}...")
                                    break
                            else:
                                print(f"⚠️  Skipped trial due to plotting error")
                                
                        except Exception as e:
                            print(f"❌ Error plotting Block {block_num}, Trial {trial_num}: {e}")
                            continue
                
                print(f"✅ Completed all plots for {config['name']}")
                
            except Exception as e:
                print(f"❌ Error analyzing {config['name']}: {e}")
                import traceback
                traceback.print_exc()

def main():
    """Main function"""
    
    # Configuration
    EEG_DATA_PATH = r"H:\Post\7th Phase (EMG-EEG Based BCI)\Stage 3_EMG Measurement\2025.06.04\Recorded EMG"
    
    # Run configurations
    run_configs = [
        {
            'eeg_file': 'WithTask_Index',
            'name': 'Participant_5_Index'
        },
        {
            'eeg_file': 'WithTask_Thumb',
            'name': 'Participant_6_Thumb'
        }
    ]
    
    # Create plotter
    plotter = InteractiveEMGPlotter(EEG_DATA_PATH)
    
    # Run interactive analysis
    plotter.run_interactive_analysis(run_configs)
    
    print(f"\n{'='*60}")
    print("🎉 INTERACTIVE ANALYSIS COMPLETE")
    print("🔧 All fixes from debugging version successfully applied!")
    print("💾 CSV files with detailed results saved for each participant!")
    print("📊 Check your EMG data folder for the CSV files!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()