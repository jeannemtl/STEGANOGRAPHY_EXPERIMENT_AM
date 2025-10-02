import numpy as np
import matplotlib.pyplot as plt
import json
from typing import List, Dict, Tuple
import random

class TrueAMSteganographyGenerator:
    """Generate reasoning chains using true amplitude modulation of linguistic features"""
    
    def __init__(self, carrier_freq=3, modulation_depth=0.6):
        """
        Args:
            carrier_freq: Carrier wave frequency (cycles per step). Period = 1/freq
            modulation_depth: How much the signal varies (0-1)
        """
        self.carrier_freq = carrier_freq
        self.modulation_depth = modulation_depth
        
        # Reasoning components for sentence construction
        self.transitions = ['Therefore', 'Thus', 'Hence', 'Consequently', 'Subsequently']
        self.subjects = ['we', 'I', 'one']
        self.verbs = ['analyze', 'examine', 'consider', 'evaluate', 'investigate']
        self.objects = ['the problem', 'the equation', 'this system', 'the data', 'these variables']
        
        # Filler words for length control
        self.fillers = [
            'carefully', 'thoroughly', 'systematically', 'precisely', 'methodically',
            'considering', 'examining', 'evaluating', 'determining', 'assessing',
            'all', 'various', 'different', 'multiple', 'several', 'numerous',
            'key', 'important', 'critical', 'essential', 'fundamental',
            'aspects', 'factors', 'elements', 'components', 'parameters'
        ]
    
    def string_to_binary(self, message: str) -> List[int]:
        """Convert string to binary representation"""
        binary = []
        for char in message:
            binary.extend([int(b) for b in format(ord(char), '08b')])
        return binary
    
    def generate_am_signal(self, message: str, num_steps: int = 100) -> Tuple[List[float], List[int]]:
        """
        Generate AM signal with smooth envelope for clean sidebands
        
        Returns:
            am_signal: Amplitude values for each step
            binary_data: Binary representation of message
        """
        binary_data = self.string_to_binary(message)
        
        # Envelope frequency - much slower than carrier
        envelope_freq = 0.05  # 20 steps per cycle for clean sidebands
        
        am_signal = []
        for step in range(num_steps):
            # Carrier wave (fast oscillation)
            carrier = np.cos(2 * np.pi * step / self.carrier_freq)
            
            # Smooth sinusoidal envelope (not binary switching)
            envelope = 1.0 + self.modulation_depth * np.cos(2 * np.pi * step * envelope_freq)
            
            # True AM signal = carrier * envelope
            amplitude = carrier * envelope
            
            am_signal.append(amplitude)
        
        return am_signal, binary_data
    
    def amplitude_to_sentence_length(self, amplitude: float) -> int:
        """
        Map amplitude to target sentence length
        
        Amplitude range: [-1, 1]
        Sentence length: [8, 22] words
        """
        # Normalize to [0, 1]
        normalized = (amplitude + 1) / 2
        
        # Map to word count
        min_length = 8
        max_length = 22
        target_length = int(min_length + normalized * (max_length - min_length))
        
        return target_length
    
    def generate_sentence_with_length(self, target_length: int, step_num: int) -> str:
        """
        Generate a reasoning sentence with exactly target_length words
        """
        # Start with base structure
        words = [
            f"In step {step_num}",  # 3 words
            random.choice(self.subjects),  # 1 word
            random.choice(self.verbs),  # 1 word
            random.choice(self.objects)  # 2-3 words
        ]
        
        current_length = len(' '.join(words).split())
        
        # Add fillers to reach target length
        remaining = target_length - current_length - 1  # -1 for ending
        
        if remaining > 0:
            # Insert fillers before the last element (object)
            filler_words = random.sample(self.fillers, min(remaining, len(self.fillers)))
            words = words[:-1] + filler_words + [words[-1]]
        
        # Add ending
        words.append(random.choice(['to obtain the solution', 'and proceed forward', 
                                    'with clear reasoning', 'for better understanding']))
        
        sentence = ' '.join(words)
        
        # Verify length and adjust if needed
        actual_words = sentence.split()
        if len(actual_words) < target_length:
            # Add more fillers
            insert_pos = len(actual_words) - 2
            while len(actual_words) < target_length:
                actual_words.insert(insert_pos, random.choice(self.fillers))
        elif len(actual_words) > target_length:
            # Remove fillers
            actual_words = actual_words[:target_length]
        
        return ' '.join(actual_words) + '.'
    
    def generate_steganographic_chain(self, message: str, num_steps: int = 40) -> Dict:
        """
        Generate complete reasoning chain with AM steganography
        
        Returns:
            Dictionary containing reasoning steps and metadata
        """
        print(f"\nGenerating AM steganographic chain...")
        print(f"Message: '{message}' ({len(message)} chars, {len(message)*8} bits)")
        print(f"Carrier frequency: {self.carrier_freq} cycles/step (period = {1/self.carrier_freq:.2f} steps)")
        
        # Generate AM signal
        am_signal, binary_data = self.generate_am_signal(message, num_steps)
        
        # Generate reasoning steps
        reasoning_steps = []
        target_lengths = []
        actual_lengths = []
        
        for step_idx, amplitude in enumerate(am_signal):
            # Determine target sentence length
            target_length = self.amplitude_to_sentence_length(amplitude)
            target_lengths.append(target_length)
            
            # Generate sentence
            sentence = self.generate_sentence_with_length(target_length, step_idx + 1)
            reasoning_steps.append(sentence)
            
            # Measure actual length
            actual_length = len(sentence.split())
            actual_lengths.append(actual_length)
        
        # Calculate correlation between target and actual
        correlation = np.corrcoef(target_lengths, actual_lengths)[0, 1]
        print(f"Target/Actual correlation: {correlation:.3f}")
        
        return {
            'message': message,
            'reasoning_steps': reasoning_steps,
            'am_signal': am_signal,
            'binary_data': binary_data,
            'target_lengths': target_lengths,
            'actual_lengths': actual_lengths,
            'carrier_freq': self.carrier_freq,
            'num_steps': num_steps,
            'metadata': {
                'length_correlation': float(correlation),
                'min_length': min(actual_lengths),
                'max_length': max(actual_lengths),
                'mean_length': np.mean(actual_lengths)
            }
        }
    
    def verify_encoding(self, result: Dict) -> Dict:
        """
        Verify that the carrier frequency is detectable
        """
        actual_lengths = np.array(result['actual_lengths'])
        
        # Normalize signal
        signal = (actual_lengths - np.mean(actual_lengths)) / (np.std(actual_lengths) + 1e-10)
        
        # FFT
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), d=1.0)
        
        # Find peak in positive frequencies
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_magnitudes = np.abs(fft_result[pos_mask])
        
        peak_idx = np.argmax(pos_magnitudes)
        detected_freq = pos_freqs[peak_idx]
        peak_power = pos_magnitudes[peak_idx]
        
        expected_freq = 1.0 / self.carrier_freq
        error = abs(detected_freq - expected_freq)
        
        verification = {
            'expected_frequency': expected_freq,
            'detected_frequency': float(detected_freq),
            'frequency_error': float(error),
            'peak_power': float(peak_power),
            'encoding_successful': error < 0.1  # Within 10% tolerance
        }
        
        return verification
    
    def visualize_encoding(self, result: Dict, save_path: str = None):
        """Create visualization of AM encoding"""
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        steps = np.arange(len(result['am_signal']))
        
        # 1. AM Signal
        axes[0].plot(steps, result['am_signal'], 'b-', linewidth=2, label='AM Signal')
        axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'AM Steganographic Signal (Message: "{result["message"]}")')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # 2. Sentence Lengths
        axes[1].plot(steps, result['target_lengths'], 'r--', linewidth=2, 
                    label='Target Length', alpha=0.7)
        axes[1].plot(steps, result['actual_lengths'], 'g-', linewidth=2, 
                    label='Actual Length')
        axes[1].set_ylabel('Words per Sentence')
        axes[1].set_title('Sentence Length Modulation')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # 3. Frequency Spectrum
        signal = np.array(result['actual_lengths'])
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), d=1.0)
        
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_magnitudes = np.abs(fft_result[pos_mask])
        
        axes[2].plot(pos_freqs, pos_magnitudes, 'b-', linewidth=2)
        axes[2].axvline(x=1/self.carrier_freq, color='r', linestyle='--', 
                       linewidth=2, label=f'Expected: {1/self.carrier_freq:.3f} cycles/step')
        axes[2].set_xlabel('Frequency (cycles/step)')
        axes[2].set_ylabel('Magnitude')
        axes[2].set_title('Frequency Spectrum (FFT)')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        axes[2].set_xlim(0, 0.5)
        
        # 4. Binary Message
        binary_display = result['binary_data'][:min(len(result['binary_data']), 100)]
        axes[3].step(range(len(binary_display)), binary_display, 'r-', linewidth=2, where='mid')
        axes[3].set_ylabel('Bit Value')
        axes[3].set_xlabel('Bit Position')
        axes[3].set_title('Encoded Binary Message (first 100 bits)')
        axes[3].set_ylim(-0.1, 1.1)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization: {save_path}")
        
        plt.show()

def main():
    """Demonstration of true AM steganography"""
    
    print("=" * 60)
    print("TRUE AMPLITUDE MODULATION STEGANOGRAPHY")
    print("=" * 60)
    
    # Initialize generator
    generator = TrueAMSteganographyGenerator(
        carrier_freq=3,  # 3 cycles per step (period = 0.33 steps)
        modulation_depth=0.6
    )
    
    # Test messages
    test_messages = ["HELLO", "SECRET", "AI_RISK"]
    
    results = []
    
    for message in test_messages:
        print(f"\n{'=' * 60}")
        print(f"ENCODING: {message}")
        print(f"{'=' * 60}")
        
        # Generate steganographic chain
        result = generator.generate_steganographic_chain(message, num_steps=40)
        
        # Verify encoding
        verification = generator.verify_encoding(result)
        
        print(f"\nVerification:")
        print(f"  Expected frequency: {verification['expected_frequency']:.3f} cycles/step")
        print(f"  Detected frequency: {verification['detected_frequency']:.3f} cycles/step")
        print(f"  Frequency error: {verification['frequency_error']:.3f}")
        print(f"  Encoding successful: {verification['encoding_successful']}")
        
        # Show first few steps
        print(f"\nFirst 3 reasoning steps:")
        for i in range(min(3, len(result['reasoning_steps']))):
            step = result['reasoning_steps'][i]
            length = result['actual_lengths'][i]
            print(f"  Step {i+1} ({length} words): {step}")
        
        # Visualize
        generator.visualize_encoding(result, f'true_am_{message.lower()}_encoding.png')
        
        result['verification'] = verification
        results.append(result)
    
    # Save all results
    # In the main() function, change this line:
    output = {
        'generator_config': {
            'carrier_freq': generator.carrier_freq,
            'modulation_depth': generator.modulation_depth
        },
        'results': [{
            'message': r['message'],
            'reasoning_steps': r['reasoning_steps'],
            'actual_lengths': r['actual_lengths'],
            'carrier_freq': r['carrier_freq'],
            'verification': {
                'expected_frequency': r['verification']['expected_frequency'],
                'detected_frequency': r['verification']['detected_frequency'],
                'frequency_error': r['verification']['frequency_error'],
                'peak_power': r['verification']['peak_power'],
                'encoding_successful': bool(r['verification']['encoding_successful'])  # Convert numpy.bool_ to bool
            }
        } for r in results]
    }
    
    with open('true_am_steganographic_data.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Generated {len(results)} true AM steganographic chains")
    print(f"Expected carrier frequency: {1/generator.carrier_freq:.3f} cycles/step")
    print(f"\nData saved to: true_am_steganographic_data.json")
    print(f"Visualizations saved as: true_am_<message>_encoding.png")

if __name__ == "__main__":
    main()
