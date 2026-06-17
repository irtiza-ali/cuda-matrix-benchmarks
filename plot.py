import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results.csv')

# --- Plot 1: Performance (Run Times) ---
plt.figure(figsize=(15, 5))

# Multiplication
plt.subplot(1, 3, 1)
plt.plot(df['Size'], df['CPU_Mul'], marker='o', label='CPU (Eigen)')
plt.plot(df['Size'], df['GPU_Mul'], marker='s', label='GPU (cuBLAS)')
plt.yscale('log')
plt.xscale('log')
plt.title('Matrix Multiplication Time')
plt.xlabel('Matrix Size (N)')
plt.ylabel('Time (ms)')
plt.legend()
plt.grid(True, which="both", ls="--")

# Inversion
plt.subplot(1, 3, 2)
plt.plot(df['Size'], df['CPU_Inv'], marker='o', label='CPU (Eigen)')
plt.plot(df['Size'], df['GPU_Inv'], marker='s', label='GPU (cuSOLVER)')
plt.yscale('log')
plt.xscale('log')
plt.title('Matrix Inversion Time')
plt.xlabel('Matrix Size (N)')
plt.legend()
plt.grid(True, which="both", ls="--")

# Eigenvalues
plt.subplot(1, 3, 3)
plt.plot(df['Size'], df['CPU_Eig'], marker='o', label='CPU (Eigen)')
plt.plot(df['Size'], df['GPU_Eig'], marker='s', label='GPU (cuSOLVER)')
plt.yscale('log')
plt.xscale('log')
plt.title('Eigenvalue Decomposition Time')
plt.xlabel('Matrix Size (N)')
plt.legend()
plt.grid(True, which="both", ls="--")

plt.tight_layout()
plt.savefig('runtime_comparison.png', dpi=300)
plt.close()

# --- Plot 2: Cumulative Precision Loss ---
plt.figure(figsize=(8, 6))
plt.plot(df['Size'], df['CPU_Loss'], marker='o', label='CPU FP32 Drift', color='blue')
plt.plot(df['Size'], df['GPU_Loss'], marker='s', label='GPU FP32 Drift (Tensor Cores/FMA)', color='red')
plt.yscale('log')
plt.xscale('log')
plt.title('Cumulative Data Loss (50 Successive Multiply/Invert Ops)\nLower is better')
plt.xlabel('Matrix Size (N)')
plt.ylabel('Frobenius Norm Error: ||X - I||')
plt.legend()
plt.grid(True, which="both", ls="--")

plt.tight_layout()
plt.savefig('precision_comparison.png', dpi=300)
plt.close()

print("Graphs generated successfully as 'runtime_comparison.png' and 'precision_comparison.png'.")