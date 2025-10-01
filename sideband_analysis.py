import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.fft import fft, fftfreq

def analyze_am_sidebands(data_file='true_am_steganographic_data.json'):
    """Analyze AM sidebands to verify true amplitude modulation"""
    
    with open(data_file, 'r') as f:
        am_data = json.load(f)
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    carrier_freq = am_data['generator_config']['carrier_freq']
    expected_carrier_freq = 1.0 / carrier_freq  # 0.333 cycles/step for carrier_freq=3
    
    for idx, result in enumerate(am_data['results']):
        message = result['message']
        actual_lengths = np.array(result['actual_lengths'])
        
        # Normalize signal
        signal = (actual_lengths - np.mean(actual_lengths)) / (np.std(actual_lengths) + 1e-10)
        
        # FFT
        fft_result = fft(signal)
        freqs = fftfreq(len(signal), d=1.0)
        
        # Positive frequencies only
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_magnitudes = np.abs(fft_result[pos_mask])
        
        # Plot frequency spectrum
        ax = axes[idx, 0]
        ax.plot(pos_freqs, pos_magnitudes, 'b-', linewidth=2)
        
        # Mark carrier frequency
        ax.axvline(x=expected_carrier_freq, color='red', linestyle='--', 
                  linewidth=2, label=f'Carrier: {expected_carrier_freq:.3f}')
        
        # Estimate modulation frequency from binary data
        binary_data = result['verification']  # This doesn't have binary, need to recalculate
        num_steps = len(actual_lengths)
        
        # For AM, modulation frequency should be much lower than carrier
        # Look for peaks below carrier frequency (envelope frequency)
        low_freq_mask = pos_freqs < expected_carrier_freq * 0.5
        if np.any(low_freq_mask):
            low_freqs = pos_freqs[low_freq_mask]
            low_mags = pos_magnitudes[low_freq_mask]
            if len(low_mags) > 0:
                modulation_peak_idx = np.argmax(low_mags)
                modulation_freq = low_freqs[modulation_peak_idx]
                
                # Calculate expected sideband positions
                upper_sideband = expected_carrier_freq + modulation_freq
                lower_sideband = expected_carrier_freq - modulation_freq
                
                ax.axvline(x=modulation_freq, color='green', linestyle=':', 
                          linewidth=2, label=f'Modulation: {modulation_freq:.3f}')
                ax.axvline(x=upper_sideband, color='orange', linestyle=':', 
                          linewidth=2, label=f'Upper SB: {upper_sideband:.3f}')
                ax.axvline(x=lower_sideband, color='purple', linestyle=':', 
                          linewidth=2, label=f'Lower SB: {lower_sideband:.3f}')
        
        ax.set_xlim(0, 0.5)
        ax.set_xlabel('Frequency (cycles/step)')
        ax.set_ylabel('Magnitude')
        ax.set_title(f'{message}: Frequency Spectrum with Sidebands')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Zoomed view around carrier
        ax = axes[idx, 1]
        zoom_range = 0.15
        zoom_mask = (pos_freqs > expected_carrier_freq - zoom_range) & (pos_freqs < expected_carrier_freq + zoom_range)
        
        if np.any(zoom_mask):
            ax.plot(pos_freqs[zoom_mask], pos_magnitudes[zoom_mask], 'b-', linewidth=2)
            ax.axvline(x=expected_carrier_freq, color='red', linestyle='--', linewidth=2)
            
            # Mark sidebands if detected
            if np.any(low_freq_mask):
                ax.axvline(x=upper_sideband, color='orange', linestyle=':', linewidth=2)
                ax.axvline(x=lower_sideband, color='purple', linestyle=':', linewidth=2)
        
        ax.set_xlabel('Frequency (cycles/step)')
        ax.set_ylabel('Magnitude')
        ax.set_title(f'{message}: Zoomed Carrier Region')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('am_sideband_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: am_sideband_analysis.png")
    plt.show()

def theoretical_am_comparison():
    """Create theoretical AM signal to show expected sideband structure"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Parameters
    carrier_freq = 3  # cycles per step
    modulation_freq = 0.05  # Much slower envelope
    num_steps = 40
    
    t = np.arange(num_steps)
    
    # Theoretical AM signal
    carrier = np.cos(2 * np.pi * t / carrier_freq)
    modulation = 1 + 0.5 * np.cos(2 * np.pi * t * modulation_freq)
    am_signal = carrier * modulation
    
    # Plot time domain
    axes[0, 0].plot(t, carrier, 'b--', alpha=0.5, label='Carrier')
    axes[0, 0].plot(t, modulation, 'g--', alpha=0.5, label='Envelope')
    axes[0, 0].plot(t, am_signal, 'r-', linewidth=2, label='AM Signal')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Theoretical AM Signal (Time Domain)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # FFT of AM signal
    fft_result = fft(am_signal)
    freqs = fftfreq(num_steps, d=1.0)
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_mags = np.abs(fft_result[pos_mask])
    
    axes[0, 1].plot(pos_freqs, pos_mags, 'b-', linewidth=2)
    axes[0, 1].axvline(x=1/carrier_freq, color='red', linestyle='--', 
                      linewidth=2, label=f'Carrier: {1/carrier_freq:.3f}')
    axes[0, 1].axvline(x=1/carrier_freq + modulation_freq, color='orange', 
                      linestyle=':', linewidth=2, label='Upper Sideband')
    axes[0, 1].axvline(x=1/carrier_freq - modulation_freq, color='purple', 
                      linestyle=':', linewidth=2, label='Lower Sideband')
    axes[0, 1].set_xlim(0, 0.5)
    axes[0, 1].set_xlabel('Frequency (cycles/step)')
    axes[0, 1].set_ylabel('Magnitude')
    axes[0, 1].set_title('Theoretical AM Spectrum (Frequency Domain)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Your actual data
    with open('true_am_steganographic_data.json', 'r') as f:
        am_data = json.load(f)
    
    result = am_data['results'][0]  # First message
    actual_lengths = np.array(result['actual_lengths'])
    signal = (actual_lengths - np.mean(actual_lengths)) / (np.std(actual_lengths) + 1e-10)
    
    axes[1, 0].plot(actual_lengths, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Sentence Length (words)')
    axes[1, 0].set_title(f'Actual AM Signal: {result["message"]} (Time Domain)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # FFT of actual
    fft_actual = fft(signal)
    freqs_actual = fftfreq(len(signal), d=1.0)
    pos_mask_actual = freqs_actual > 0
    pos_freqs_actual = freqs_actual[pos_mask_actual]
    pos_mags_actual = np.abs(fft_actual[pos_mask_actual])
    
    axes[1, 1].plot(pos_freqs_actual, pos_mags_actual, 'g-', linewidth=2)
    axes[1, 1].axvline(x=1/3, color='red', linestyle='--', linewidth=2, label='Expected Carrier')
    axes[1, 1].set_xlim(0, 0.5)
    axes[1, 1].set_xlabel('Frequency (cycles/step)')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].set_title(f'Actual Spectrum: {result["message"]}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('theoretical_vs_actual_am.png', dpi=300, bbox_inches='tight')
    print("Saved: theoretical_vs_actual_am.png")
    plt.show()

if __name__ == "__main__":
    print("=== AM SIDEBAND ANALYSIS ===\n")
    analyze_am_sidebands()
    theoretical_am_comparison()
    print("\nAnalysis complete!")
