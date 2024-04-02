from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import Estimator as AerEstimator
import matplotlib.pyplot as plt
import numpy as np


# Define the Hamiltonian terms
E_1 = 0.
E_2 = 4.
V_11 = 3.
V_12 = .2
V_22 = -V_11
eps = .5*(E_1+E_2)
sgm = .5*(E_1-E_2)
c = .5*(V_11+V_22)
omg_z = .5*(V_11-V_22)
omg_x = V_12

# Define the Pauli matrices and the identity matrix
X = Operator(Pauli('X')).data
Y = Operator(Pauli('Y')).data
Z = Operator(Pauli('Z')).data
I = Operator(Pauli('I')).data

H0 = eps*I + sgm*Z
H1 = c*I + omg_z*Z + omg_x*X
num_lambdas_ = 50
lambdas_ = np.linspace(0, 1, num=num_lambdas_)

# Define the Hamiltonian terms
h11 = [c + eps, [0,1], [I]]
h12 = [sgm, [0], [Z]]
h21 = [omg_z, [0], [Z]]
h22 = [omg_x, [0], [X]]
H = [h11, h12, h21, h22]


def eigsys(H0, H1, lambdas_):
    """
    Compute the eigenvalues and eigenvectors of a Hamiltonian for a range of lambda values.

    Parameters:
    H0 (np.array): The base Hamiltonian.
    H1 (np.array): The Hamiltonian to be added for each lambda.
    lambdas_ (np.array): The range of lambda values.

    Returns:
    tuple: The eigenvalues and eigenvectors of the Hamiltonian for each lambda.
    """

    Hamil = np.array([H0 + lbd * H1 for lbd in lambdas_])
    eVals = np.array([np.linalg.eigvalsh(H) for H in Hamil])
    eVecs = np.array([np.linalg.eigh(H)[1].flatten() for H in Hamil])
    return eVals, eVecs

eVals, eVecs = eigsys(H0, H1, lambdas_)

npMES = NumPyMinimumEigensolver()
lambd_num = np.linspace(0, 1, num=10)

def npEigs(lambd_num):
    """
    Compute the minimum eigenvalue of a Hamiltonian for a range of lambda values using NumPy.

    Parameters:
    lambd_num (np.array): The range of lambda values.

    Returns:
    list: The minimum eigenvalue of the Hamiltonian for each lambda.
    """

    npEigs = [npMES.compute_minimum_eigenvalue(
        SparsePauliOp.from_list([("I", eps + lbd*c), ("X", lbd*omg_x), ("Z", sgm + lbd*omg_z)])
    ).eigenvalue.real for lbd in lambd_num]
    return npEigs

npEigsComp = npEigs(lambd_num)

# VQE setup and execution
nEst = AerEstimator(run_options={"shots": 1024})
iterations = 125
ansatz = TwoLocal(1, rotation_blocks=["rx", "ry"], reps=0)

def vqe(lambdas_, estimator, ansatz):
    """
    Compute the minimum eigenvalue of a Hamiltonian for a range of lambda values using VQE.

    Parameters:
    lambdas_ (np.array): The range of lambda values.
    estimator (AerEstimator): The estimator to use in the VQE algorithm.
    ansatz (TwoLocal): The ansatz to use in the VQE algorithm.

    Returns:
    list: The minimum eigenvalue of the Hamiltonian for each lambda.
    """

    optimizer = SPSA(maxiter=iterations)
    vqe = VQE(estimator, ansatz, optimizer=optimizer)
    results = [vqe.compute_minimum_eigenvalue(
        SparsePauliOp.from_list([("I", eps + lbd*c), ("X", lbd*omg_x), ("Z", sgm + lbd*omg_z)])
    ).eigenvalue.real for lbd in lambdas_]
    return results

VQERes = vqe(lambd_num, nEst, ansatz)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(lambdas_, eVals[:, 0], label=r'$\epsilon_0$')
ax.plot(lambdas_, eVals[:, 1], label=r'$\epsilon_1$')
ax.plot(lambd_num, npEigsComp, 'o', label='NumPyMinimumEigensolver')
ax.plot(lambd_num, VQERes, '--', label='Qiskit_VQE')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('E')
ax.legend()
plt.grid(True)
fig.tight_layout()
fig.savefig('/home/odinjo/Project_1/QML_project_1/figures/figs_part_c/eigvals_qiskitVSnumpy.png', format='png')
