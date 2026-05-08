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
    ├── mat_100k_KR_p_arm.txt        # GM12878 Chr7 (0-58.1Mb) at 100kb resolution (KR normalized)
    └── mat_50k_KR_chr7_23_27Mb.txt  # High-resolution sub-matrix of Chr7 (23-27Mb) at 50kb (KR normalized)
```
## ⚙️ Requirements
CHORD is developed using Python 3.11 and requires minimal external dependencies.
Please ensure the following packages are installed in your environment:

* numpy

* scikit-learn

* matplotlib

* seaborn

* tqdm

## 🚀 Usage
CHORD requires a normalized intra-chromosomal Hi-C contact matrix as input (tab or space-separated pure numeric text file, without headers or bin annotations).
