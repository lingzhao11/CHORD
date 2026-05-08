# CHORD: Deterministic Demarcation of Topologically Associating Domains via Exact Optimization

CHORD is a robust computational framework that fundamentally shifts the paradigm of TAD detection from heuristic approximation to exact mathematical solving. By rigorously optimizing the Calinski-Harabasz (CH) index via dynamic programming and 2D prefix sum acceleration, CHORD guarantees the identification of global optima, ensuring highly stable performance even under extreme data sparsity and noise.

![CHORD Framework](data/main.pdf) *(Note: Upload your image to the data folder and name it framework.jpg)*

## 📂 Repository Structure

The repository is kept ultra-lightweight and highly integrated.

```text
CHORD/
├── chord.py              # The core algorithm and visualization script (All-in-one)
├── requirements.txt      # Python dependencies
└── data/                 # Sample datasets for testing
    ├── example_0_58Mb.txt    # 100kb resolution Hi-C matrix (0-58.1Mb)
    └── local_4Mb.txt         # High-resolution sub-matrix (e.g., chr7:23-27Mb)
