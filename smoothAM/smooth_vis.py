import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft, fftfreq
import json

def create_3d_sideband_visualization(data_file='true_am_steganographic_data.json'):
    """3D visualization focused on carrier and sidebands"""
    
    with open(data_file, 'r') as f:
        am_data = json.load(f)
    
    # Create ONE figure with 3x3 grid
    fig = plt.figure(figsize=(20, 14))
    
    carrier_freq = am_data['generator_config']['carrier_freq']
    expected_carrier = 1.0 / carrier_freq
    
    messages = []
    all_spectra = []
    
    for idx, result in enumerate(am_data['results']):
        message = result['message']
        messages.append(message)
        actual_lengths = np.array(result['actual_lengths'])
        
        # High-pass filter to remove DC and low frequencies
        signal = (actual_lengths - np.mean(actual_lengths)) / (np.std(actual_lengths) + 1e-10)
        fft_result = fft(signal)
        freqs = fftfreq(len(signal), d=1.0)
        
        # Focus on carrier region only (0.15 to 0.5 cycles/step)
        focus_mask = (freqs > 0.15) & (freqs < 0.5)
        focus_freqs = freqs[focus_mask]
        focus_magnitudes = np.abs(fft_result[focus_mask])
        
        all_spectra.append((focus_freqs, focus_magnitudes))
    
    # Subplot positions in 3x3 grid
    ax1 = fig.add_subplot(3, 3, (1, 2), projection='3d')  # Row 1, cols 1-2
    
    X = all_spectra[0][0]
    Y = np.arange(len(messages))
    Z = np.array([spectrum[1] for spectrum in all_spectra])
    
    X_grid, Y_grid = np.meshgrid(X, Y)
    
    # Plot with better color mapping
    surf = ax1.plot_surface(X_grid, Y_grid, Z, cmap='plasma', 
                           alpha=0.9, edgecolor='k', linewidth=0.5)
    
    # Mark sideband positions
    envelope_freq = 0.05
    lower_sideband = expected_carrier - envelope_freq
    upper_sideband = expected_carrier + envelope_freq
    
    ax1.set_xlabel('Frequency (cycles/step)', fontsize=12)
    ax1.set_ylabel('Message', fontsize=12)
    ax1.set_zlabel('Magnitude', fontsize=12)
    ax1.set_title('Carrier Region: Clean Sidebands', fontsize=14, fontweight='bold')
    ax1.set_yticks(Y)
    ax1.set_yticklabels(messages)
    ax1.view_init(elev=30, azim=225)
    
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # 2D overlay
    ax2 = fig.add_subplot(3, 3, 3)
    
    for idx, (freqs, mags) in enumerate(all_spectra):
        ax2.plot(freqs, mags, linewidth=2, label=messages[idx], alpha=0.8)
    
    ax2.axvline(x=lower_sideband, color='purple', linestyle='--', 
               linewidth=2, label=f'Lower SB: {lower_sideband:.3f}')
    ax2.axvline(x=expected_carrier, color='red', linestyle='--', 
               linewidth=2, label=f'Carrier: {expected_carrier:.3f}')
    ax2.axvline(x=upper_sideband, color='orange', linestyle='--', 
               linewidth=2, label=f'Upper SB: {upper_sideband:.3f}')
    
    ax2.set_xlabel('Frequency (cycles/step)')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Sideband Structure (2D)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Waterfall
    ax3 = fig.add_subplot(3, 3, (4, 5))
    
    offset = 0
    for idx, (freqs, mags) in enumerate(all_spectra):
        ax3.fill_between(freqs, offset, offset + mags, alpha=0.7, 
                        label=messages[idx])
        offset += max(mags) * 1.3
    
    ax3.axvline(x=lower_sideband, color='purple', linestyle=':', linewidth=2)
    ax3.axvline(x=expected_carrier, color='red', linestyle='--', linewidth=2)
    ax3.axvline(x=upper_sideband, color='orange', linestyle=':', linewidth=2)
    
    ax3.set_xlabel('Frequency (cycles/step)')
    ax3.set_ylabel('Magnitude (offset)')
    ax3.set_title('Waterfall: Carrier Region')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Individual traces - Row 3: positions 7, 8, 9
    for i in range(3):
        ax = fig.add_subplot(3, 3, 7 + i)
        freqs, mags = all_spectra[i]
        
        ax.plot(freqs, mags, 'b-', linewidth=2)
        ax.axvline(x=lower_sideband, color='purple', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(x=expected_carrier, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(x=upper_sideband, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax.set_title(f'{messages[i]}: Three-Peak Structure')
        ax.set_xlabel('Frequency (cycles/step)')
        ax.set_ylabel('Magnitude')
        ax.grid(True, alpha=0.3)
    
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
