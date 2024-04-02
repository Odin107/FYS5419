import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qiskit.visualization import plot_state_qsphere, plot_bloch_multivector, plot_state_city
from qiskit.quantum_info import partial_trace, DensityMatrix
"""
# Define constants and setup
lambda_ = 0.4
eps_list = [0, 2.5, 6.5, 7]
H_0 = np.diag(eps_list)  # Simplified diagonal matrix creation

pauli_x = np.array([[0, 1], [1, 0]])
pauli_z = np.array([[1, 0], [0, -1]])
H_x = 2.0
H_z = 3.0
H_I = H_x * np.kron(pauli_x, pauli_x) + H_z * np.kron(pauli_z, pauli_z)

H_tot = H_0 + lambda_ * H_I

# Diagonalize the total Hamiltonian
eig_vals, eig_vecs = np.linalg.eigh(H_tot)

# Sort eigenvalues and eigenvectors
sorted_indices = eig_vals.argsort()
eig_vals = eig_vals[sorted_indices]
eig_vecs = eig_vecs[:, sorted_indices]

# Compute the density matrix for the ground state
DM = np.outer(eig_vecs[:, 0], np.conj(eig_vecs[:, 0]))

# Compute reduced density matrices for subsystems A and B
rho_0 = DensityMatrix(DM)
rho_A = partial_trace(rho_0, [1])
rho_B = partial_trace(rho_0, [0])

# Function to calculate von Neumann entropy
def von_neumann_entropy(rho):
    eigenvalues = np.linalg.eigvalsh(rho.data)
    return -np.sum(eigenvalues * np.log2(eigenvalues + 1e-15))  # Adding a small value to avoid log(0)

# Compute entropies
S_A = von_neumann_entropy(rho_A)
S_B = von_neumann_entropy(rho_B)

print(f"Von Neumann Entropy for subsystem A: {S_A}")
print(f"Von Neumann Entropy for subsystem B: {S_B}")

# Visualize the state
plot_state_qsphere(rho_0).savefig('/home/odinjo/Project_1/QML_project_1/figures/figs_part_d/qsphere_rho_0.png')
plot_bloch_multivector(rho_0).savefig('/home/odinjo/Project_1/QML_project_1/figures/figs_part_d/bloch_rho_0.png')
plot_state_city(rho_0.data, title='Density Matrix').savefig('/home/odinjo/Project_1/QML_project_1/figures/figs_part_d/city_rho_0.png')
"""

# Constants
lambda_ = 0.4
eps_list = [0, 2.5, 6.5, 7]
H_x = 2.0
H_z = 3.0

# State vectors
s_0 = np.array([1, 0])
s_1 = np.array([0, 1])

# Hamiltonians
H_0 = np.diag(eps_list)  # Diagonal matrix with eps_list as diagonal elements
pauli_x = np.array([[0, 1], [1, 0]])
pauli_z = np.array([[1, 0], [0, -1]])
H_I = H_x * np.kron(pauli_x, pauli_x) + H_z * np.kron(pauli_z, pauli_z)
H_tot = H_0 + lambda_ * H_I

# Eigenvalues and eigenvectors
eig_vals, eig_vecs = np.linalg.eigh(H_tot)

# Density matrix
rho_d = np.dot(eig_vecs * eig_vecs.conj().T, eig_vecs * eig_vecs.conj().T)

print(rho_d)

# Calculating the partial density matrices and entropies
tol = 1e-3
rho_a = np.outer(s_0, s_0) * (np.sum(np.square(eig_vecs[:2, 0:2]), axis=0)) + \
        np.outer(s_1, s_1) * (np.sum(np.square(eig_vecs[:2, 2:4]), axis=0))
rho_b = np.outer(s_0, s_0) * (np.sum(np.square(eig_vecs[0, :2]), axis=0) + np.sum(np.square(eig_vecs[2, :2]), axis=0)) + \
        np.outer(s_1, s_1) * (np.sum(np.square(eig_vecs[1, :2]), axis=0) + np.sum(np.square(eig_vecs[3, :2]), axis=0))

S_a = -np.trace(rho_a) * np.log(np.trace(rho_a + tol))
S_b = -np.trace(rho_b) * np.log(np.trace(rho_b) + tol)

print(S_a, S_b)


# Constants
lambda_ = 0.4

# Assuming H_0 and H_I are defined elsewhere in your code
H_tot = H_0 + lambda_ * H_I

# Eigenvalues and eigenvectors for H_0, H_I, and H_tot
eig_vals0, eig_vecs0 = np.linalg.eigh(H_0)
eig_vals_i, eig_vecs_i = np.linalg.eigh(lambda_ * H_I)
eig_vals, eig_vecs = np.linalg.eigh(H_tot)

# Density matrix using the ground state (eigenvector corresponding to the lowest eigenvalue)
DM = np.outer(eig_vecs[:, 0], eig_vecs[:, 0].conj())

# Projection operations
d = np.eye(2)
v1 = np.array([1, 0])
proj1 = np.kron(v1, d)
x1 = proj1 @ DM @ proj1.T

v2 = np.array([0, 1])
proj2 = np.kron(v2, d)
x2 = proj2 @ DM @ proj2.T


plot_state_qsphere(DM).savefig('/home/odinjo/Project_1/QML_project_1/figures/figs_part_d/qsphere_rho_0.png')
plot_bloch_multivector(DM, title=f'Bloch Spheres for density matrix given $\lambda$= {lambda_}').savefig('/home/odinjo/Project_1/QML_project_1/figures/figs_part_d/bloch_rho_0.png')
plot_state_city(x1, title=f'Reduced density matrix  for $ \lambda$: {lambda_}').savefig('/home/odinjo/Project_1/QML_project_1/figures/figs_part_d/x1_city_rho_0.png')
plot_state_city(x2, title=f'Reduced density Matrix  for $\lambda$: {lambda_}').savefig('/home/odinjo/Project_1/QML_project_1/figures/figs_part_d/x2_city_rho_0.png')


def entropy(lambda_):
    H_tot = H_0 + lambda_ * H_I
    eig_vals, eig_vecs = np.linalg.eigh(H_tot)
    
    eig_vecs_reshaped = eig_vecs.reshape((2, 2, -1), order='F')
    

    rho_a = np.zeros((2, 2), dtype=np.complex128)
    rho_b = np.zeros((2, 2), dtype=np.complex128)
    for i in range(eig_vecs.shape[1]):  
        vec = eig_vecs[:, i].reshape((2, 2), order='F')
        rho_a += np.outer(vec[:, 0], vec[:, 0].conj())
        rho_b += np.outer(vec[0, :], vec[0, :].conj())

    rho_a /= np.trace(rho_a)
    rho_b /= np.trace(rho_b)

  
    epsilon = 1e-10
    rho_a_diag = np.diag(rho_a) + epsilon
    rho_b_diag = np.diag(rho_b) + epsilon

    S_a = -np.sum(rho_a_diag * np.log2(rho_a_diag))
    S_b = -np.sum(rho_b_diag * np.log2(rho_b_diag))
    
    return S_a.real, S_b.real  

# Example usage
lambda_list = np.linspace(0, 2, 100)
rho_a_list = []
rho_b_list = []
for lambda_ in lambda_list:
    S_a, S_b = entropy(lambda_)
    rho_a_list.append(S_a)
    rho_b_list.append(S_b)

    
plt.figure(figsize=(10, 6))
plt.plot(lambda_list, rho_a_list, label='$S_{a}$')
plt.plot(lambda_list, rho_b_list, label='$S_{b}$', linestyle='--')
plt.title('Entropy as a function of connection strength')
plt.xlabel('$\lambda$')
plt.ylabel('Entropy')
plt.legend()
plt.grid(True)
plt.savefig('/home/odinjo/Project_1/QML_project_1/figures/figs_part_d/entropy_vs_lambda.png')



"""
import numpy as np
import matplotlib.pyplot as plt

# Constants
lambda_ = 0.4  # Consistent lambda value for the entire script
eps_list = [0, 2.5, 6.5, 7]
H_x = 2.0
H_z = 3.0

# State vectors
s_0 = np.array([1, 0])
s_1 = np.array([0, 1])

# Hamiltonians
H_0 = np.diag(eps_list)  # Diagonal matrix with eps_list as diagonal elements
pauli_x = np.array([[0, 1], [1, 0]])
pauli_z = np.array([[1, 0], [0, -1]])
H_I = H_x * np.kron(pauli_x, pauli_x) + H_z * np.kron(pauli_z, pauli_z)
H_tot = H_0 + lambda_ * H_I

# Eigenvalues and eigenvectors
eig_vals, eig_vecs = np.linalg.eigh(H_tot)  # Use eigh for Hermitian matrices

# Function to compute entropy of a density matrix
def entropy_of_density_matrix(rho):

    rho += 1e-10 * np.eye(rho.shape[0])  # Add a small value to the diagonal to avoid log(0)
    rho_diag = np.diag(rho)
    return -np.sum(rho_diag * np.log2(rho_diag))

# Function to compute the entropy of the system given lambda
def system_entropy(lambda_):
    H_tot = H_0 + lambda_ * H_I
    eig_vals, eig_vecs = np.linalg.eigh(H_tot)
    ground_state = eig_vecs[:, np.argmin(eig_vals)]
    rho = np.outer(ground_state, ground_state.conj())
    return entropy_of_density_matrix(rho)

# Example usage: Compute and plot entropy as a function of lambda
lambda_list = np.linspace(0, 2, 100)
entropy_list = [system_entropy(lmd) for lmd in lambda_list]

plt.figure(figsize=(10, 6))
plt.plot(lambda_list, entropy_list, label='Entropy')
plt.title('Entropy as a function of $\lambda$')
plt.xlabel('$\lambda$')
plt.ylabel('Entropy')
plt.legend()
plt.grid(True)
plt.savefig('/home/odinjo/Project_1/QML_project_1/figures/figs_part_d/entropy_vs_lambda.png')
"""










