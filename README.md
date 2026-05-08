# CHORD: Deterministic Demarcation of Topologically Associating Domains via Exact Optimization

CHORD is a robust computational framework that fundamentally shifts the paradigm of TAD detection from heuristic approximation to exact mathematical solving. By rigorously optimizing the Calinski-Harabasz (CH) index via dynamic programming and 2D prefix sum acceleration, CHORD guarantees the identification of global optima, ensuring highly stable performance even under extreme data sparsity and noise.

![CHORD Framework](data/framework.jpg)

## 📂 Repository Structure

The repository is kept ultra-lightweight and highly integrated.

```text
CHORD/
├── chord.py                         # The core algorithm and visualization script (All-in-one)
└── data/                            # Sample datasets for testing
    ├── simulated_data.txt           # Simulated Hi-C matrix for algorithmic validation
    ├── mat_100k_KR_p_arm.txt        # GM12878 Chr7 (0-58.1Mb) at 100kb resolution
    └── mat_50k_KR_chr7_23_27Mb.txt  # High-resolution sub-matrix of Chr7 (23-27Mb) at 50kb
