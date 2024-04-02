import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SPSA
from qiskit_aer.primitives import Estimator as AerEstimator
from Part_d import entropy

def setup_hamil(dim, energies, Hx, Hz):
    """
    Set up the Hamiltonian matrices H0 and HI.

    Parameters:
    dim (int): The dimension of the Hamiltonian matrices.
    energies (np.array): The energies for the diagonal elements of H0.
    Hx (float): The coefficient for the X interaction term in HI.
    Hz (float): The coefficient for the Z interaction term in HI.

    Returns:
    tuple: The Hamiltonian matrices H0 and HI.
    """

    H0 = np.diag(energies)
    X, Z = np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, -1]])
    HI = Hx * np.kron(X, X) + Hz * np.kron(Z, Z)
    return H0, HI

def comp_eVals_and_entropies(lbds, H0, HI):
    """
    Compute the eigenvalues and entropies for a range of lambda values.

    Parameters:
    lbds (np.array): The range of lambda values.
    H0 (np.array): The H0 Hamiltonian matrix.
    HI (np.array): The HI Hamiltonian matrix.

    Returns:
    tuple: The eigenvalues and entropies for each lambda value in lbds.
    """

    eigvals,  entropies = [], []
    for lbd in lbds:
        Hamiltonian = H0 + lbd * HI
        eig_vals, eig_vects = np.linalg.eigh(Hamiltonian)
        eigvals.append(eig_vals)
        entropies.append(entropy(eig_vects, 0))  # Assuming entropy function works with vector input
    return np.array(eigvals), np.array(entropies)

def setup_and_run_vqe(lbds, E0, E1, E2, E3, Hx, Hz):
    """
    Set up and run the VQE algorithm for a range of lambda values.

    Parameters:
    lbds (np.array): The range of lambda values.
    E0, E1, E2, E3 (float): The coefficients for the terms in the observable.
    Hx (float): The coefficient for the X interaction term in the observable.
    Hz (float): The coefficient for the Z interaction term in the observable.

    Returns:
    list: The minimum eigenvalues computed by the VQE algorithm for each lambda value in lbds.
    """
    results = []
    for lbd in lbds:
        observable = SparsePauliOp.from_list([
            ("II", (E0 + E1 + E2 + E3) / 4),
            ("ZZ", (E0 - E1 - E2 + E3) / 4 + lbd * Hz),
            ("ZI", (E0 + E1 - E2 - E3) / 4),
            ("IZ", (E0 - E1 + E2 - E3) / 4),
            ("XX", lbd * Hx)])
        result = vqe.compute_minimum_eigenvalue(operator=observable).eigenvalue.real
        results.append(result)
    return results

def plot_results(lbds_eig, eigvals, lbds_numpy, numpy_eigs, lbds_vqe, VQE_res):
    """
    Plot the eigenvalues and VQE results for a range of lambda values.

    Parameters:
    lbds_eig (np.array): The range of lambda values for the eigenvalues.
    eigvals (np.array): The eigenvalues for each lambda value in lbds_eig.
    lbds_numpy (np.array): The range of lambda values for the numpy eigenvalues.
    numpy_eigs (np.array): The eigenvalues computed by the numpy eigensolver for each lambda value in lbds_numpy.
    lbds_vqe (np.array): The range of lambda values for the VQE results.
    VQE_res (list): The minimum eigenvalues computed by the VQE algorithm for each lambda value in lbds_vqe.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    for i in range(eigvals.shape[1]):
        plt.plot(lbds_eig, eigvals[:, i], label=f'$\\epsilon_{i}$')
    plt.plot(lbds_numpy, numpy_eigs, 'r--', label='Numpy min eigensolver')
    plt.plot(lbds_vqe, VQE_res, 'gx', label='Qiskit VQE')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Energy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('part_e_eigvals.png', format='png')

dim, energies = 4, [0.0, 2.5, 6.5, 7.0]
H0, HI = setup_hamil(dim, energies, 2.0, 3.0)
lbds = np.linspace(0, 1, num=50)

eigvals, entropies = comp_eVals_and_entropies(lbds, H0, HI)

numpy_solver = NumPyMinimumEigensolver()
lbds_num = np.linspace(0, 1, num=10)
numpy_eigs = [numpy_solver.compute_minimum_eigenvalue(SparsePauliOp.from_list([
    ("II", (sum(energies)) / 4),
    ("ZZ", (energies[0] - energies[1] - energies[2] + energies[3]) / 4 + lbd * 3.0),
    ("ZI", (energies[0] + energies[1] - energies[2] - energies[3]) / 4),
    ("IZ", (energies[0] - energies[1] + energies[2] - energies[3]) / 4),
    ("XX", lbd * 2.0)])).eigenvalue.real for lbd in lbds_num]

ansatz = TwoLocal(2, rotation_blocks=["rx", "ry"], entanglement_blocks='cx', reps=1)
spsa = SPSA(maxiter=400)
vqe = VQE(AerEstimator(run_options={"shots": 1024}), ansatz, optimizer=spsa)
lbds_vqe = np.linspace(0, 1, num=10)  
VQE_res = setup_and_run_vqe(lbds_vqe, *energies, 2.0, 3.0)

plot_results(lbds, eigvals, lbds_num, numpy_eigs, lbds_vqe, VQE_res)
