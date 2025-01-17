import numpy as np
import matplotlib.pyplot as plt

def fvm_1d_diffusion(L, N, T_A, T_B, k, A, S):
    """
    Solves the 1D steady-state diffusion equation using the finite volume method.

    Parameters:
    - L (float): Length of the domain (m).
    - N (int): Number of control volumes (cells).
    - T_A (float): Temperature at the left boundary (C).
    - T_B (float): Temperature at the right boundary (C).
    - k (float): Thermal conductivity (W/mK).
    - A (float): Cross-sectional area (m^2).
    - S (float): Heat source (W/m^3).

    Returns:
    - x (ndarray): Cell center positions (m).
    - T (ndarray): Temperature at each cell center (C).
    """
    # Mesh generation
    dx = L / N  # Length of each cell
    x = np.linspace(dx / 2, L - dx / 2, N)  # Cell center positions

    # Initialize coefficient arrays
    a_P = np.zeros(N)  # Main diagonal (aP)
    a_L = np.zeros(N)  # Lower diagonal (aL)
    a_R = np.zeros(N)  # Upper diagonal (aR)
    b = np.zeros(N)    # Right-hand side (Su)

    # Coefficients for the interior cells
    D = k * A / dx  # Diffusive conductance
    for i in range(1, N - 1):
        a_L[i] = D
        a_R[i] = D
        a_P[i] = a_L[i] + a_R[i]
        b[i] = S * A * dx

    # Left boundary cell
    a_L[0] = 0
    a_R[0] = D
    a_P[0] = a_L[0] + a_R[0] - (-2 * D)
    b[0] = S * A * dx + 2 * D * T_A

    # Right boundary cell
    a_L[N - 1] = D
    a_R[N - 1] = 0
    a_P[N - 1] = a_L[N - 1] + a_R[N - 1] - (-2 * D)
    b[N - 1] = S * A * dx + 2 * D * T_B

    # Assemble the sparse tridiagonal matrix
    A_matrix = np.zeros((N, N))
    for i in range(N):
        A_matrix[i, i] = a_P[i]
        if i > 0:
            A_matrix[i, i - 1] = -a_L[i]
        if i < N - 1:
            A_matrix[i, i + 1] = -a_R[i]

    # Solve the linear system
    T = np.linalg.solve(A_matrix, b)

    return x, T

# Example usage
if __name__ == "__main__":
    # Parameters
    L = 5.0         # Length of the bar (m)
    N = 100         # Number of cells
    T_A = 100.0     # Left boundary temperature (C)
    T_B = 200.0     # Right boundary temperature (C)
    k = 100.0       # Thermal conductivity (W/mK)
    A = 0.1         # Cross-sectional area (m^2)
    S = 1000.0      # Heat source (W/m^3)

    # Solve
    x, T = fvm_1d_diffusion(L, N, T_A, T_B, k, A, S)

    # Analytical solution
    T_analytical = T_A + (T_B - T_A) * x / L + (S * (x * (L - x))) / (2 * k)

    # Plot results
    plt.plot(x, T, label="FVM Solution", marker="o")
    plt.plot(x, T_analytical, label="Analytical Solution", linestyle="--")
    plt.xlabel("Position (m)")
    plt.ylabel("Temperature (C)")
    plt.title("1D Steady-State Heat Diffusion")
    plt.legend()
    plt.grid()
    plt.show()
