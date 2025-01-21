import numpy as np
import matplotlib.pyplot as plt

def fvm_1d_convection_diffusion(L, N, T_A, T_B, k, A, S, rho, c_p, U):
    """
    Solves the 1D steady-state convection-diffusion equation using the finite volume method.

    Parameters:
    - L (float): Length of the domain (m).
    - N (int): Number of control volumes (cells).
    - T_A (float): Temperature at the left boundary (C).
    - T_B (float): Temperature at the right boundary (C).
    - k (float): Thermal conductivity (W/mK).
    - A (float): Cross-sectional area (m^2).
    - S (float): Heat source (W/m^3).
    - rho (float): Density of the fluid (kg/m^3).
    - c_p (float): Specific heat capacity (J/kgK).
    - U (float): Flow velocity (m/s).

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

    # Diffusive and convective fluxes
    D = k * A / dx  # Diffusive conductance
    F = rho * c_p * U * A  # Convective conductance

    # Coefficients for the interior cells
    for i in range(1, N - 1):
        a_L[i] = D + F / 2
        a_R[i] = D - F / 2
        a_P[i] = a_L[i] + a_R[i]
        b[i] = S * A * dx

    # Left boundary cell
    a_L[0] = 0
    a_R[0] = D - F / 2
    a_P[0] = a_R[0] + (F) + 2 * D
    b[0] = S * A * dx + 2 * D * T_A

    # Right boundary cell
    a_L[N - 1] = D + F / 2
    a_R[N - 1] = 0
    a_P[N - 1] = a_L[N - 1] - (F) + 2 * D
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
    N = 20         # Number of cells
    T_A = 100.0     # Left boundary temperature (C)
    T_B = 200.0     # Right boundary temperature (C)
    k = 100.0       # Thermal conductivity (W/mK)
    A = 0.1         # Cross-sectional area (m^2)
    S = 1000.0      # Heat source (W/m^3)
    rho = 1.0       # Fluid density (kg/m^3)
    c_p = 1000.0    # Specific heat capacity (J/kgK)
    U = 3         # Flow velocity (m/s)

    # Solve
    x, T = fvm_1d_convection_diffusion(L, N, T_A, T_B, k, A, S, rho, c_p, U)

    # Plot results
    plt.plot(x, T, label="FVM Solution", marker="o")
    plt.xlabel("Position (m)")
    plt.ylabel("Temperature (C)")
    plt.title("1D Steady-State Convection-Diffusion")
    plt.legend()
    plt.grid()
    plt.show()
