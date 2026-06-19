import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("results.csv")

# Set up the figure for Runtime Comparison
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("CPU vs GPU Runtimes for Naive Linear Algebra Algorithms", fontsize=16)

# 1. Multiplication
axs[0].plot(df['N'], df['CPU_Mult_ms'], marker='o', label='CPU Mult', color='red')
axs[0].plot(df['N'], df['GPU_Mult_ms'], marker='s', label='GPU Mult', color='blue')
axs[0].set_title("Matrix Multiplication O(N³)")
axs[0].set_xlabel("Matrix Size (N)")
axs[0].set_ylabel("Time (ms)")
axs[0].set_xscale('log', base=2)
axs[0].set_yscale('log')
axs[0].grid(True, which="both", ls="--", alpha=0.5)
axs[0].legend()

# 2. Inversion
axs[1].plot(df['N'], df['CPU_Inv_ms'], marker='o', label='CPU Gauss-Jordan', color='red')
axs[1].plot(df['N'], df['GPU_Inv_ms'], marker='s', label='GPU Gauss-Jordan', color='blue')
axs[1].set_title("Matrix Inversion (Gauss-Jordan)")
axs[1].set_xlabel("Matrix Size (N)")
axs[1].set_yscale('log')
axs[1].set_xscale('log', base=2)
axs[1].grid(True, which="both", ls="--", alpha=0.5)
axs[1].legend()

# 3. Eigenvalues
axs[2].plot(df['N'], df['CPU_Eigen_ms'], marker='o', label='CPU Power Iteration', color='red')
axs[2].plot(df['N'], df['GPU_Eigen_ms'], marker='s', label='GPU Power Iteration', color='blue')
axs[2].set_title("Dominant Eigenvalue (Power Iteration)")
axs[2].set_xlabel("Matrix Size (N)")
axs[2].set_yscale('log')
axs[2].set_xscale('log', base=2)
axs[2].grid(True, which="both", ls="--", alpha=0.5)
axs[2].legend()

plt.tight_layout()
plt.savefig("runtime_comparison.png", dpi=300)
print("Saved runtime_comparison.png")

# Set up the figure for Precision Comparison
plt.figure(figsize=(8, 6))
plt.plot(df['N'], df['CPU_Error'], marker='o', label='CPU Precision Error', color='red')
plt.plot(df['N'], df['GPU_Error'], marker='s', label='GPU Precision Error', color='blue')
plt.title("Precision Loss Comparison: || A * A_inv - I ||_F")
plt.xlabel("Matrix Size (N)")
plt.ylabel("Frobenius Norm Error (Log Scale)")
plt.xscale('log', base=2)
plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("precision_comparison.png", dpi=300)
print("Saved precision_comparison.png")