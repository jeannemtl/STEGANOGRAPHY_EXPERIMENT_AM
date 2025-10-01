# Frequency-Domain Steganography Detection

Research comparing frequency domain based steganographic encoding with neural representation clustering.

## Scripts

### `true_am_generator.py`
Generates reasoning chains with frequency-domain encoding.
- Carrier: 3 cycles/step (0.33 cycles/step period)
- Modulates sentence length (8-22 words) based on binary message
- **Output**: `true_am_steganographic_data.json` + visualization PNGs

### `sideband_analysis.py`
Analyzes frequency spectrum for AM sideband structure.
- Compares theoretical AM vs actual implementation
- Reveals quasi-AM: dominant carrier with degraded sidebands
- **Output**: `am_sideband_analysis.png`, `theoretical_vs_actual_am.png`

### `vis.py`
3D PCA visualization of BERT representations across 11 steganographic patterns.
- Compares: Normal, Step/Now, First/Then, Word Length, Punctuation, Template AM (3), True AM (3)
- **Key result**: Template and continuous AM cluster similarly, but separably with different frequency signatures
- **Output**: `comprehensive_steganographic_analysis.png`, evolution GIF

**Conclusion**: BERT encodes sentence-level complexity. Frequency-domain encoding shows sidebands in frequency-domain analysis. The differences are reported as separable using PCA. Unclear for what reason.

## Installation
```bash
pip install torch transformers numpy matplotlib scipy scikit-learn pillow
```

## Usage
```bash
python3 true_am_generator.py      # Generate data
python3 sideband_analysis.py      # Analyze spectrum
python3 vis.py           # Visualize BERT clustering
```
