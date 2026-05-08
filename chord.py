import numpy as np
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from tqdm import tqdm

class CHORD:
    def __init__(self, mat):
        """
        Initialize the CHORD model.
        """
        # Validate matrix: Only non-negative values are accepted
        if (mat < 0).any():
            raise ValueError("Invalid matrix given: negative values observed.")
        
        # Copy matrix and remove diagonal interactions
        self.mat = mat.copy()
        np.fill_diagonal(self.mat, 0)
        self.n_bins = self.mat.shape[0]

        # Dynamically set max target number of TADs to 1/3 of the total bins
        self.max_k = max(2, self.n_bins // 3)
        self.optimal_k = None

        # Initialize data containers for dynamic programming
        self.hicmat_avesum = np.zeros((self.n_bins, self.n_bins))
        self.dp_table = np.zeros((self.n_bins, self.max_k))
        self.boundary_candidates = np.zeros((self.n_bins, self.max_k))

    def _cal_avesum(self):
        """
        Calculate the average sum of each TAD in the Hi-C matrix.
        """
        bin_sum = np.zeros((self.n_bins, self.n_bins))
        ave_value = np.average(self.mat, axis=0)
        norm_mat = self.mat - ave_value

        # Compute cumulative sums using norm_mat
        for start in range(self.n_bins):
            if start == 0:
                bin_sum[start, :] = norm_mat[start, :]
            else:
                bin_sum[start, :] = norm_mat[start, :] + bin_sum[start - 1, :]
                
        # Calculate the variance-like metric for each possible TAD block
        for start in range(self.n_bins):
            for end in range(start, self.n_bins, 1):
                if start == 0:
                    current_matrix = bin_sum[end, :] / (end + 1)
                else:
                    current_matrix = (bin_sum[end, :] - bin_sum[start - 1, :]) / (end - start + 1)
                
                self.hicmat_avesum[start, end] = np.sum(current_matrix ** 2)

    def fit(self):
        """
        Main solver using dynamic programming to populate the optimal split tables.
        """
        self._cal_avesum()
        
        for cluster in tqdm(range(self.max_k), desc="DP Optimization", unit="k"):
            for end in range(cluster, self.n_bins, 1):
                if cluster == 0:
                    self.dp_table[end, cluster] = self.hicmat_avesum[0, end] * (end + 1)
                    self.boundary_candidates[end, cluster] = 0
                else:
                    for i in range(0, end, 1):
                        if i >= cluster - 1:
                            temp = self.hicmat_avesum[i + 1, end] * (end - i) + self.dp_table[i, cluster - 1]
                            if temp > self.dp_table[end, cluster]:
                                self.dp_table[end, cluster] = temp
                                self.boundary_candidates[end, cluster] = i

    def _get_boundaries(self, k):
        """
        Backtrace the dynamic programming table to extract breakpoints for a specific K.
        """
        c_idx = k - 1  # 0-based cluster index
        boundaries = []
        i = self.n_bins - 1
        
        while i >= 0:
            boundaries.append(i)
            if c_idx < 0:
                break
            next_i = int(self.boundary_candidates[i, c_idx])
            if i == next_i:
                break
            i = next_i
            c_idx -= 1
            
        # Ensure 0 is added to cap the leftmost boundary
        if len(boundaries) > 0 and boundaries[-1] != 0:
            boundaries.append(0)
            
        return np.unique(np.sort(boundaries))

    def find_optimal_k(self):
        """
        Evaluate all possible K values using the Silhouette Score and return the best one.
        """
        # Create distance matrix (prune=1e-6 to avoid division by zero)
        distance_arr = 1 / (self.mat + 1e-6)
        np.fill_diagonal(distance_arr, 0)

        best_k = 2
        best_score = -1.0

        for k in range(2, self.max_k + 1):
            boundaries = self._get_boundaries(k)
            if len(boundaries) < 2:
                continue

            # Create labels for silhouette score evaluation
            labels = np.zeros(self.n_bins)
            for idx in range(len(boundaries) - 1):
                start = boundaries[idx]
                end = boundaries[idx + 1]
                if idx > 0:
                    start += 1
                labels[int(start):int(end) + 1] = idx

            try:
                score = silhouette_score(distance_arr, labels, metric="precomputed")
                if score > best_score:
                    best_score = score
                    best_k = k
            except ValueError:
                pass

        self.optimal_k = best_k
        return self.optimal_k

    def get_tad_domains(self, force_k=None):
        """
        Convert breakpoints into a list of 1-based, Start-End TAD pairs.
        If force_k is provided, bypass the Silhouette Score selection.
        """
        if force_k is not None:
            self.optimal_k = force_k
        elif self.optimal_k is None:
            self.find_optimal_k()
            
        boundaries = self._get_boundaries(self.optimal_k)
        domains = []
        
        if len(boundaries) < 2:
            return domains
            
        for idx in range(len(boundaries) - 1):
            start = boundaries[idx]
            end = boundaries[idx + 1]
            
            if idx > 0:
                start += 1
                
            domains.append((int(start) + 1, int(end) + 1))
            
        return domains


# --- Plotting Module ---

def squeeze_outliers(mat, percent=99):
    percentile = np.percentile(mat.flatten(), percent)
    op_percentile = np.percentile(mat.flatten(), 100 - percent)
    mat_copy = mat.copy()
    mat_copy[mat_copy > percentile] = percentile
    mat_copy[mat_copy < op_percentile] = op_percentile
    return mat_copy

def plot_tad_heatmap(matrix, boundaries, method_name="CHORD", output_file="tad_heatmap.png", cmap="Reds"):
    fig, ax = plt.subplots(figsize=(12, 10))

    mat_plot = squeeze_outliers(matrix)
    np.fill_diagonal(mat_plot, 0) 
    
    sns.heatmap(mat_plot, cmap=cmap, linewidths=0, linecolor=None, ax=ax, cbar_kws={'shrink': 0.8})
    ax.axis('off')
    
    cax = fig.axes[-1]
    cax.tick_params(labelsize=18)
    
    if boundaries:
        for i in range(len(boundaries)):
            start = int(boundaries[i][0])
            end = int(boundaries[i][1])
            x = np.arange(start - 1, end + 1)
            y1 = (start - 1) * np.ones(len(x))
            y2 = (end) * np.ones(len(x))
            
            box_color = '#0072B2' 
            line_w = 1.5          
            alpha_val = 0.8 

            ax.plot(x, y1, color=box_color, linewidth=line_w, alpha=alpha_val)
            ax.plot(y2, x, color=box_color, linewidth=line_w, alpha=alpha_val)
            ax.plot(y1, x, color=box_color, linewidth=line_w, alpha=alpha_val)
            ax.plot(x, y2, color=box_color, linewidth=line_w, alpha=alpha_val)
    
    ax.set_title(method_name, fontsize=36, fontweight='bold', pad=20) 
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


# --- Main Command Line Execution ---

def main():
    parser = argparse.ArgumentParser(description="CHORD: Deterministic Demarcation of TADs via Exact Optimization")
    parser.add_argument("--matrix", dest="matrix", type=str, required=True, help="Path to the input Hi-C matrix file")
    parser.add_argument("--k", dest="k", type=int, default=None, help="Force a specific number of TADs (bypasses automatic Silhouette selection)")
    parser.add_argument("--out_boundary", dest="out_boundary", type=str, default="result.txt", help="Output filename for the TAD boundaries")
    parser.add_argument("--out_image", dest="out_image", type=str, default="tad_heatmap.png", help="Output filename for the heatmap image")
    parser.add_argument("--cmap", dest="cmap", type=str, default="Reds", help="Colormap for the heatmap")
    
    args = parser.parse_args()

    # Load matrix with error handling
    try:
        print(f"Loading matrix from: {args.matrix} ...")
        mat = np.loadtxt(args.matrix)
    except Exception as e:
        print(f"\n[ERROR] Failed to load matrix: {e}")
        print("Please ensure your input is a pure numeric text file (tab or space separated) without headers, strings, or missing values.")
        return

    print("Running DP Solver...")
    mod = CHORD(mat)
    mod.fit()

    if args.k:
        print(f"Extracting domains for user-defined K={args.k} (bypassing automatic selection)...")
    else:
        print("Evaluating optimal K using Silhouette Score...")
        
    tads = mod.get_tad_domains(force_k=args.k)
    
    # Write results to output file (Bin indexing only)
    with open(args.out_boundary, "w") as f:
        f.write("Start_Bin\tEnd_Bin\n") 
        for start_bin, end_bin in tads:
            f.write(f"{start_bin}\t{end_bin}\n")
            
    print(f"Successfully saved {len(tads)} domains to {args.out_boundary}")

    # Plot the results
    print("Plotting heatmap with bounding boxes...")
    plot_tad_heatmap(matrix=mat, boundaries=tads, method_name="CHORD", output_file=args.out_image, cmap=args.cmap)
    print(f"Heatmap saved to {args.out_image}")
    print("Done!")

if __name__ == '__main__':
    main()