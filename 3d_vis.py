import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft, fftfreq
import json

def create_3d_sideband_visualization(data_file='true_am_steganographic_data.json'):
    """3D visualization of frequency spectrum across all messages"""
    
    with open(data_file, 'r') as f:
        am_data = json.load(f)
    
    fig = plt.figure(figsize=(16, 12))
    
    # Main 3D surface plot
    ax1 = fig.add_subplot(2, 2, (1, 2), projection='3d')
    
    carrier_freq = am_data['generator_config']['carrier_freq']
    expected_carrier = 1.0 / carrier_freq
    
    messages = []
    all_spectra = []
    
    for idx, result in enumerate(am_data['results']):
        message = result['message']
        messages.append(message)
        actual_lengths = np.array(result['actual_lengths'])
        
        # Normalize and compute FFT
        signal = (actual_lengths - np.mean(actual_lengths)) / (np.std(actual_lengths) + 1e-10)
        fft_result = fft(signal)
        freqs = fftfreq(len(signal), d=1.0)
        
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_magnitudes = np.abs(fft_result[pos_mask])
        
        all_spectra.append((pos_freqs, pos_magnitudes))
    
    # Create 3D surface
    X = all_spectra[0][0]  # Frequencies (same for all)
    Y = np.arange(len(messages))  # Message index
    Z = np.array([spectrum[1] for spectrum in all_spectra])
    
    X_grid, Y_grid = np.meshgrid(X, Y)
    
    # Plot surface
    surf = ax1.plot_surface(X_grid, Y_grid, Z, cmap='viridis', 
                           alpha=0.8, edgecolor='none')
    
    # Mark carrier frequency with red plane
    carrier_plane = np.ones_like(Z) * expected_carrier
    ax1.plot_surface(carrier_plane, Y_grid, Z * 0.3, color='red', 
                     alpha=0.3, label='Carrier')
    
    # Plot individual traces
    for idx, (freqs, mags) in enumerate(all_spectra):
        ax1.plot(freqs, [idx]*len(freqs), mags, 'k-', linewidth=1, alpha=0.5)
    
    ax1.set_xlabel('Frequency (cycles/step)', fontsize=12)
    ax1.set_ylabel('Message', fontsize=12)
    ax1.set_zlabel('Magnitude', fontsize=12)
    ax1.set_title('3D Frequency Spectrum: Carrier and Sidebands', fontsize=14, fontweight='bold')
    ax1.set_yticks(Y)
    ax1.set_yticklabels(messages)
    ax1.set_xlim(0, 0.5)
    ax1.view_init(elev=25, azim=45)
    
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # Waterfall plot
    ax2 = fig.add_subplot(2, 2, 3)
    
    offset = 0
    for idx, (freqs, mags) in enumerate(all_spectra):
        ax2.fill_between(freqs, offset, offset + mags, alpha=0.6, 
                        label=messages[idx])
        offset += max(mags) * 1.2
    
    ax2.axvline(x=expected_carrier, color='red', linestyle='--', 
               linewidth=2, label='Expected Carrier')
    ax2.set_xlabel('Frequency (cycles/step)')
    ax2.set_ylabel('Magnitude (offset)')
    ax2.set_title('Waterfall Plot: Stacked Spectra')
    ax2.set_xlim(0, 0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Sideband detail zoom
    ax3 = fig.add_subplot(2, 2, 4)
    
    zoom_range = 0.15
    for idx, (freqs, mags) in enumerate(all_spectra):
        mask = (freqs > expected_carrier - zoom_range) & (freqs < expected_carrier + zoom_range)
        ax3.plot(freqs[mask], mags[mask], linewidth=2, label=messages[idx], alpha=0.8)
    
    # Mark theoretical sideband positions
    modulation_freq = 0.075  # Estimated from spectrum
    ax3.axvline(x=expected_carrier, color='red', linestyle='--', linewidth=2, label='Carrier')
    ax3.axvline(x=expected_carrier + modulation_freq, color='orange', 
               linestyle=':', linewidth=2, label='Upper SB')
    ax3.axvline(x=expected_carrier - modulation_freq, color='purple', 
               linestyle=':', linewidth=2, label='Lower SB')
    
    ax3.set_xlabel('Frequency (cycles/step)')
    ax3.set_ylabel('Magnitude')
    ax3.set_title('Zoomed: Carrier Region with Sidebands')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('3d_sideband_visualization.png', dpi=300, bbox_inches='tight')
    print("Saved: 3d_sideband_visualization.png")
    plt.show()

def create_rotating_3d_animation():
    """Create animated rotating 3D spectrum"""
    from matplotlib.animation import FuncAnimation, PillowWriter
    
    with open('true_am_steganographic_data.json', 'r') as f:
        am_data = json.load(f)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare data
    messages = []
    all_spectra = []
    
    for result in am_data['results']:
        messages.append(result['message'])
        actual_lengths = np.array(result['actual_lengths'])
        signal = (actual_lengths - np.mean(actual_lengths)) / (np.std(actual_lengths) + 1e-10)
        fft_result = fft(signal)
        freqs = fftfreq(len(signal), d=1.0)
        pos_mask = freqs > 0
        all_spectra.append((freqs[pos_mask], np.abs(fft_result[pos_mask])))
    
    X = all_spectra[0][0]
    Y = np.arange(len(messages))
    Z = np.array([spectrum[1] for spectrum in all_spectra])
    X_grid, Y_grid = np.meshgrid(X, Y)
    
    def animate(frame):
        ax.clear()
        surf = ax.plot_surface(X_grid, Y_grid, Z, cmap='viridis', 
                              alpha=0.8, edgecolor='none')
        
        carrier_freq = 1.0 / am_data['generator_config']['carrier_freq']
        carrier_plane = np.ones_like(Z) * carrier_freq
        ax.plot_surface(carrier_plane, Y_grid, Z * 0.3, color='red', alpha=0.3)
        
        ax.set_xlabel('Frequency (cycles/step)')
        ax.set_ylabel('Message')
        ax.set_zlabel('Magnitude')
        ax.set_title('3D Frequency Spectrum with Carrier')
        ax.set_yticks(Y)
        ax.set_yticklabels(messages)
        ax.set_xlim(0, 0.5)
        
        # Rotate view
        ax.view_init(elev=25, azim=frame)
        
        return surf,
    
    anim = FuncAnimation(fig, animate, frames=np.arange(0, 360, 2), 
                        interval=50, blit=False)
    
    writer = PillowWriter(fps=20)
    anim.save('3d_sideband_rotation.gif', writer=writer, dpi=100)
    print("Saved: 3d_sideband_rotation.gif")
    plt.close()

if __name__ == "__main__":
    print("Creating 3D sideband visualization...")
    create_3d_sideband_visualization()
    
    print("\nCreating rotating animation...")
    create_rotating_3d_animation()
    
    print("\nDone!")
