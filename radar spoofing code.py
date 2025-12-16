import numpy as np
import matplotlib.pyplot as plt
import time

class RadarSimulator:
    def __init__(self, target_range=90, target_velocity=0, 
                 enable_spoofing=False, spoof_range=None, spoof_velocity=None):
        """
        Initialize FMCW Radar Simulator
        
        Parameters:
        - target_range: Distance to target in meters
        - target_velocity: Target velocity in m/s (positive = moving away)
        - enable_spoofing: Enable spoofing attack (default: False)
        - spoof_range: Spoofed range in meters (None = no range spoofing)
        - spoof_velocity: Spoofed velocity in m/s (None = no velocity spoofing)
        """
        # Radar parameters
        self.c = 3e8  # Speed of light (m/s)
        self.fc = 77e9  # Carrier frequency (77 GHz - automotive radar)
        self.B = 150e6  # Bandwidth (150 MHz)
        self.T_chirp = 50e-6 # Chirp duration (50 microseconds)
        self.fs = 10e6  # Sampling frequency (10 MHz)
        
        # Target parameters
        self.target_range = target_range
        self.target_velocity = target_velocity
        
        # Spoofing parameters - ADDED
        self.enable_spoofing = enable_spoofing
        self.spoof_range = spoof_range if spoof_range is not None else target_range
        self.spoof_velocity = spoof_velocity if spoof_velocity is not None else target_velocity
        
        # Derived parameters
        self.slope = self.B / self.T_chirp  # Chirp slope (Hz/s)
        self.max_range = (self.c * self.T_chirp) / 2
        self.range_resolution = self.c / (2 * self.B)
        
        # Calculate maximum unambiguous velocity
        self.lambda_wavelength = self.c / self.fc
        self.v_max = self.lambda_wavelength / (4 * self.T_chirp)
        
        # Time vector
        self.t = np.arange(0, self.T_chirp, 1/self.fs)
        
    def generate_chirp(self):
        """Generate transmitted FMCW chirp signal"""
        # Linear frequency modulation
        freq_inst = self.fc + self.slope * self.t
        phase = 2 * np.pi * (self.fc * self.t + 0.5 * self.slope * self.t**2)
        chirp = np.cos(phase)
        return chirp, freq_inst
    
    def generate_echo(self):
        """Generate received echo from target"""
        # Calculate time delay due to range
        tau = 2 * self.target_range / self.c
        
        # Calculate Doppler frequency shift
        f_doppler = 2 * self.target_velocity * self.fc / self.c
        
        # Delayed time vector
        t_delayed = self.t - tau
        t_delayed = np.maximum(t_delayed, 0)  # Causal signal
        
        # Generate echo with delay and Doppler
        phase_echo = 2 * np.pi * ((self.fc + f_doppler) * t_delayed + 
                                   0.5 * self.slope * t_delayed**2)
        
        # Add propagation loss
        wavelength = self.c / self.fc
        path_loss = (4 * np.pi * self.target_range / wavelength)**2
        amplitude = 0.1 / np.sqrt(path_loss)  # Scaling factor for visibility
        
        echo = amplitude * np.cos(phase_echo)
        freq_inst_echo = self.fc + f_doppler + self.slope * t_delayed
        
        return echo, freq_inst_echo
    
    def generate_range_doppler_map(self, num_chirps=128):
        """
        Generate Range-Doppler map using multiple chirps
        Includes spoofing signals if enabled
        
        Parameters:
        - num_chirps: Number of chirps to simulate (for Doppler processing)
        """
        # Initialize data matrix
        N_samples = len(self.t)
        data_matrix = np.zeros((N_samples, num_chirps), dtype=complex)
        
        # True target parameters
        tau_true = 2 * self.target_range / self.c
        f_beat_true = self.slope * tau_true  # Beat frequency due to range
        f_doppler_true = 2 * self.target_velocity * self.fc / self.c
        
        # Spoofed target parameters (when enabled)
        if self.enable_spoofing:
            tau_spoof = 2 * self.spoof_range / self.c
            f_beat_spoof = self.slope * tau_spoof
            f_doppler_spoof = 2 * self.spoof_velocity * self.fc / self.c
        
        for chirp_idx in range(num_chirps):
            # Time offset for this chirp (simulates motion)
            t_chirp_start = chirp_idx * self.T_chirp
            
            # True target signal 
            doppler_phase_true = 2 * np.pi * f_doppler_true * t_chirp_start
            signal_true = 1.0 * np.exp(1j * 2 * np.pi * f_beat_true * self.t + 1j * doppler_phase_true)
            
            # Spoofed signal (when enabled)
            if self.enable_spoofing:
                doppler_phase_spoof = 2 * np.pi * f_doppler_spoof * t_chirp_start
                signal_spoof = 1.0 * np.exp(1j * 2 * np.pi * f_beat_spoof * self.t + 1j * doppler_phase_spoof)
                total_signal = signal_true + signal_spoof
            else:
                total_signal = signal_true
            
            # Adding noise
            noise = 0.01 * (np.random.randn(N_samples) + 1j * np.random.randn(N_samples))
            data_matrix[:, chirp_idx] = total_signal + noise
        
        # Range FFT (across fast-time samples within each chirp)
        range_fft = np.fft.fft(data_matrix, axis=0, n=2048)  # Zero-pad for better resolution
        
        # Doppler FFT (across slow-time chirps)
        range_doppler = np.fft.fft(range_fft, axis=1)
        range_doppler = np.fft.fftshift(range_doppler, axes=1)
        
        # Magnitude in dB
        rd_map = 20 * np.log10(np.abs(range_doppler) + 1e-10)
        
        # Create range axis
        freq_range = np.fft.fftfreq(range_fft.shape[0], 1/self.fs)
        range_axis = (freq_range * self.c / (2 * self.slope))
        
        # Create velocity axis
        doppler_freq = np.fft.fftshift(np.fft.fftfreq(num_chirps, self.T_chirp))
        velocity_axis = doppler_freq * self.c / (2 * self.fc)
        
        return rd_map, range_axis, velocity_axis
    
    def generate_beat_signal_with_noise(self):
        """
        Generate beat signal from target echo with added noise
        """
        demo_range = 10
        tau = 2 * demo_range / self.c
        f_beat = self.slope * tau
        
        # Generate beat signal (oscillating at beat frequency)
        beat_signal = np.cos(2 * np.pi * f_beat * self.t)
        
        # Add noise
        noise = 0.4 * np.random.randn(len(self.t))
        noisy_signal = beat_signal + noise
        
        return noisy_signal, beat_signal, noise, demo_range

    def smooth_with_convolution(self, noisy_signal, window_size=25):
        """
        Applies smoothing convolution to reduce noise
        """
        # Create smoothing kernel (moving average)
        kernel = np.ones(window_size) / window_size
        
        # Convolution
        smoothed_signal = np.convolve(noisy_signal, kernel, mode='same')
        
        return smoothed_signal, kernel

    def plot_convolution_demo(self):
        """
        Simple 3-plot demonstration: Noisy → True Signal → Smoothed (via convolution)
        """
        # Generate signals
        noisy_signal, true_signal, noise, demo_range = self.generate_beat_signal_with_noise()
        smoothed_signal, kernel = self.smooth_with_convolution(noisy_signal, window_size=25)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Calculate SNR
        signal_power = np.mean(true_signal**2)
        noise_power = np.mean(noise**2)
        snr_before = 10 * np.log10(signal_power / noise_power)
        
        residual_noise = smoothed_signal - true_signal
        residual_power = np.mean(residual_noise**2)
        snr_after = 10 * np.log10(signal_power / residual_power)
        
        # Plot 1: Noisy Signal (Raw received signal)
        axes[0].plot(self.t * 1e6, noisy_signal, 'r', linewidth=1, alpha=0.8)
        axes[0].set_xlabel('Time (μs)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Amplitude', fontsize=11, fontweight='bold')
        axes[0].set_title('Noisy Received Signal', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([0, 30])
        axes[0].set_ylim([-2, 2])
        
        info_text = f"SNR: {snr_before:.1f} dB"
        axes[0].text(0.02, 0.98, info_text, transform=axes[0].transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9))
        
        # Plot 2: True Signal (What we want to recover)
        axes[1].plot(self.t * 1e6, true_signal, 'b', linewidth=2)
        axes[1].set_xlabel('Time (μs)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Amplitude', fontsize=11, fontweight='bold')
        axes[1].set_title('True Signal', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([0, 30])
        axes[1].set_ylim([-2, 2])
        
        beat_freq = self.slope * (2 * demo_range / self.c)
        info_text = f"Frequency: {beat_freq/1e3:.1f} kHz\nDemo target at {demo_range}m"
        axes[1].text(0.02, 0.98, info_text, transform=axes[1].transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
        
        # Plot 3: After Convolution (Smoothed signal)
        axes[2].plot(self.t * 1e6, smoothed_signal, 'g', linewidth=2, label='After Convolution')
        axes[2].plot(self.t * 1e6, true_signal, 'b--', linewidth=1.5, alpha=0.6, label='True Signal')
        axes[2].set_xlabel('Time (μs)', fontsize=11, fontweight='bold')
        axes[2].set_ylabel('Amplitude', fontsize=11, fontweight='bold')
        axes[2].set_title('After Convolution', fontsize=13, fontweight='bold', color='green')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim([0, 30])
        axes[2].set_ylim([-2, 2])
        axes[2].legend(loc='upper right', fontsize=11)
        
        improvement = snr_after - snr_before
        info_text = (f"SNR: {snr_after:.1f} dB\n"
                    f"Improvement: +{improvement:.1f} dB\n")
        axes[2].text(0.02, 0.98, info_text, transform=axes[2].transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
        
        # Add convolution formula
        formula_text = "Convolution"
        axes[2].text(0.98, 0.02, formula_text, transform=axes[2].transAxes,
                    fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        plt.suptitle('Convolution Demonstration', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.show()
        
        # Print summary
        print("\n" + "="*60)
        print("CONVOLUTION DEMONSTRATION SUMMARY")
        print("="*60)
        print(f"Demo Target Range: {demo_range} m (for visualization)")
        print(f"Beat Frequency: {beat_freq/1e3:.1f} kHz")
        print(f"\nNoise Reduction Results:")
        print(f"  SNR Before: {snr_before:.1f} dB (noisy)")
        print(f"  SNR After:  {snr_after:.1f} dB (smoothed)")
        print(f"  Improvement: +{improvement:.1f} dB")
        print(f"\nConvolution successfully recovered the signal pattern!")
        print("="*60)




    def plot_chirp_time_domain(self):
        """Plot transmitted and received chirps in time domain"""
        tx_chirp, _ = self.generate_chirp()
        rx_echo, _ = self.generate_echo()
        
        plt.figure(figsize=(14, 5))
        
        # Transmitted chirp
        plt.subplot(1, 2, 1)
        plt.plot(self.t * 1e6, tx_chirp, 'b', linewidth=0.8)
        plt.xlabel('Time (μs)', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.title('Transmitted Chirp - Time Domain', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 5])
        
        info_text = ("This shows the transmitted radar signal.\n"
                    "It's a high-frequency cosine wave whose\n"
                    "frequency increases linearly over time\n"
                    "(FMCW chirp).")
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Received echo
        plt.subplot(1, 2, 2)
        plt.plot(self.t * 1e6, rx_echo, 'r', linewidth=0.8)
        plt.xlabel('Time (μs)', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.title(f'Received Echo - Time Domain\n(Target: {self.target_range}m, {self.target_velocity}m/s)', 
                 fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 5])
        
        tau = 2 * self.target_range / self.c
        info_text = (f"This is the reflected signal from the target.\n"
                    f"It's delayed by {tau*1e9:.2f} ns due to\n"
                    f"the round-trip travel time to {self.target_range}m.\n"
                    f"The frequency is also shifted by Doppler effect.")
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        plt.tight_layout()
        plt.show()
    
    def plot_chirp_frequency(self):
        """Plot instantaneous frequency vs time"""
        tx_chirp, freq_tx = self.generate_chirp()
        rx_echo, freq_rx = self.generate_echo()
        
        plt.figure(figsize=(14, 5))
        
        # Transmitted chirp frequency
        plt.subplot(1, 2, 1)
        plt.plot(self.t * 1e6, (freq_tx - self.fc) / 1e6, 'b', linewidth=2)
        plt.xlabel('Time (μs)', fontsize=12)
        plt.ylabel('Frequency Offset from Carrier (MHz)', fontsize=12)
        plt.title('Transmitted Chirp - Frequency vs Time', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        info_text = (f"This shows how the chirp frequency sweeps\n"
                    f"linearly from 0 to {self.B/1e6:.0f} MHz over {self.T_chirp*1e6:.0f} μs.\n"
                    f"Carrier freq: {self.fc/1e9:.1f} GHz\n"
                    f"This linear sweep is what makes a radar FMCW.")
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Received echo frequency
        plt.subplot(1, 2, 2)
        plt.plot(self.t * 1e6, (freq_rx - self.fc) / 1e6, 'r', linewidth=2)
        plt.xlabel('Time (μs)', fontsize=12)
        plt.ylabel('Frequency Offset from Carrier (MHz)', fontsize=12)
        plt.title('Received Echo - Frequency vs Time', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        f_doppler = 2 * self.target_velocity * self.fc / self.c
        info_text = (f"The echo has the same linear sweep but:\n"
                    f"1. Delayed in time\n"
                    f"2. Shifted by Doppler: {f_doppler/1e3:.2f} kHz\n"
                    f"This frequency difference between TX and RX\n"
                    f"reveals the target's range and velocity.")
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        plt.tight_layout()
        plt.show()
    
    def plot_range_doppler_map(self):
        """Plot the Range-Doppler map (what the radar 'sees')"""
        rd_map, range_axis, velocity_axis = self.generate_range_doppler_map()
        
        plt.figure(figsize=(12, 8))
        
        # Limiting range and velocity values
        range_mask = (range_axis >= 0) & (range_axis <= 150)
        velocity_mask = (velocity_axis >= -50) & (velocity_axis <= 50)
        
        rd_map_plot = rd_map[np.ix_(range_mask, velocity_mask)]
        range_axis_plot = range_axis[range_mask]
        velocity_axis_plot = velocity_axis[velocity_mask]
        
        # Normalize for better visualization
        rd_map_norm = rd_map_plot - np.min(rd_map_plot)
        threshold = np.max(rd_map_norm) - 40  # Show 40 dB dynamic range
        
        im = plt.imshow(rd_map_norm.T, aspect='auto', 
                       extent=[range_axis_plot[0], range_axis_plot[-1],
                              velocity_axis_plot[0], velocity_axis_plot[-1]],
                       origin='lower', cmap='hot', vmin=threshold, 
                       interpolation='bilinear')
        
        plt.xlabel('Range (m)', fontsize=13, fontweight='bold')
        plt.ylabel('Velocity (m/s)', fontsize=13, fontweight='bold')
        
        # Title changes based on spoofing
        if self.enable_spoofing:
            plt.title('Range-Doppler Map', 
                     fontsize=14, fontweight='bold', color='red')
        else:
            plt.title('Range-Doppler Map', 
                     fontsize=14, fontweight='bold')
        
        plt.grid(True, alpha=0.3, color='cyan', linewidth=0.5, linestyle='--')
        
        # Mark true target position
        
        plt.plot(self.target_range, self.target_velocity, 'go', 
                markersize=15, markeredgewidth=3, markeredgecolor='white',
                label=f'True Target: {self.target_range}m, {self.target_velocity}m/s', zorder=10)
        
        # Mark spoofed target if enabled
        if self.enable_spoofing:
            plt.plot(self.spoof_range, self.spoof_velocity, 'r^', 
                    markersize=18, markeredgewidth=3, markeredgecolor='yellow',
                    label=f'SPOOFED: {self.spoof_range}m, {self.spoof_velocity}m/s', zorder=11)
    
        # Check for velocity aliasing
        if abs(self.target_velocity) > self.v_max:
            v_aliased = self.target_velocity % (2 * self.v_max)
            if v_aliased > self.v_max:
                v_aliased = v_aliased - 2 * self.v_max
            plt.plot(self.target_range, v_aliased, 'yo', 
                    markersize=12, alpha=0.6,
                    label=f'Aliased: {v_aliased:.1f}m/s', zorder=9)
        
        plt.legend(loc='upper right', fontsize=11)
        
        plt.colorbar(im, label='Signal Strength (dB)', pad=0.02)
        
        # Info text with spoofing details
        aliasing_warning = ""
        if abs(self.target_velocity) > self.v_max:
            aliasing_warning = f"\n Velocity aliasing on true target"
        
        spoof_info = ""
        if self.enable_spoofing:
            spoof_type = []
            if self.spoof_range != self.target_range:
                spoof_type.append("RANGE")
            if self.spoof_velocity != self.target_velocity:
                spoof_type.append("VELOCITY")
            spoof_info = f"\n\n SPOOFING: {' + '.join(spoof_type)}\n  Spoofer injects false signal"
        
        box_color = 'red' if self.enable_spoofing else 'black'
        text_color = 'yellow' if self.enable_spoofing else 'white'
        
        info_text = (f"Bright spots = detected objects\n"
                    f"Range resolution: {self.range_resolution:.2f} m\n"
                    f"Max unambiguous velocity: ±{self.v_max:.2f} m/s"
                    f"{aliasing_warning}{spoof_info}")
        '''
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8),
                color=text_color)
        '''
        plt.tight_layout()
        plt.show()
    
    def plot_all(self):
        """Generate all plots"""
        print("\n" + "="*60)
        print("FMCW RADAR SIMULATION")
        if self.enable_spoofing:
            print("SPOOFING ATTACK ACTIVE")
        print("="*60)
        print(f"\nRadar Configuration:")
        print(f"  Carrier Frequency: {self.fc/1e9:.1f} GHz")
        print(f"  Bandwidth: {self.B/1e6:.0f} MHz")
        print(f"  Chirp Duration: {self.T_chirp*1e6:.0f} μs")
        print(f"  Range Resolution: {self.range_resolution:.2f} m")
        print(f"  Max Unambiguous Velocity: ±{self.v_max:.2f} m/s")
        print(f"\nTrue Target:")
        print(f"  Range: {self.target_range} m")
        print(f"  Velocity: {self.target_velocity} m/s")
        print(f"  Time Delay: {2*self.target_range/self.c*1e9:.2f} ns")
        print(f"  Doppler Shift: {2*self.target_velocity*self.fc/self.c/1e3:.2f} kHz")
        
        # Spoofing info
        if self.enable_spoofing:
            print(f"\nSpoofed Target:")
            print(f"  Range: {self.spoof_range} m", end="")
            if self.spoof_range != self.target_range:
                print(f" (RANGE SPOOFING: Δ = {self.spoof_range - self.target_range:+.1f}m)")
            else:
                print(" (no range spoofing)")
            print(f"  Velocity: {self.spoof_velocity} m/s", end="")
            if self.spoof_velocity != self.target_velocity:
                print(f" (VELOCITY SPOOFING: Δ = {self.spoof_velocity - self.target_velocity:+.1f}m/s)")
            else:
                print(" (no velocity spoofing)")
        
        # Velocity aliasing check
        if abs(self.target_velocity) > self.v_max:
            v_aliased = self.target_velocity % (2 * self.v_max)
            if v_aliased > self.v_max:
                v_aliased = v_aliased - 2 * self.v_max
            print(f"\n WARNING: VELOCITY ALIASING on true target!")
            print(f"  Target velocity ({self.target_velocity} m/s) exceeds v_max")
            print(f"  Will appear at: {v_aliased:.2f} m/s")
        
        print("="*60 + "\n")
        
        #self.plot_chirp_time_domain()
        self.plot_chirp_frequency()
        self.plot_range_doppler_map()
        #self.plot_convolution_demo()


# Run simulation
if __name__ == "__main__":    
    start_time = time.time()
    # True target parameters
    TARGET_RANGE = 100       # meters
    TARGET_VELOCITY = 10     # m/s (positive is moving away, negative is moving closer)
    
    # Spoofing controls - ADDED
    ENABLE_SPOOFING = True   # Set to False for no spoofing
    SPOOF_RANGE = 80      # meters (None = same as true target)
    SPOOF_VELOCITY = 7.5  #m/s (None = same as true target)
    # =======================================
    
    # Creates radar simulation
    radar = RadarSimulator(
        target_range=TARGET_RANGE,
        target_velocity=TARGET_VELOCITY,
        enable_spoofing=ENABLE_SPOOFING,     
        spoof_range=SPOOF_RANGE,             
        spoof_velocity=SPOOF_VELOCITY        
    )
    end_time = time.time()
    print(f"Total simulation time: {end_time - start_time:.9f} seconds")
    # Generate all plots
    radar.plot_all()
    
    print("\nSimulation complete!")