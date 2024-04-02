import numpy as np
import matplotlib.pyplot as plt
from qiskit_aer import Aer
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms import VQE, NumPyMinimumEigensolver


def eig(E, V, W, mode):
    """
    Compute the eigenvalue for a given mode.

    Parameters:
    E (float): The energy.
    V (float): The potential.
    W (float): The interaction strength.
    mode (int): The mode (0, 1, or 2).

    Returns:
    float: The eigenvalue for the given mode.
    """
    if mode == 0:
        return np.sqrt(E**2 + V**2)
    elif mode == 1:
        return W
    elif mode == 2:
        return -np.sqrt(E**2 + V**2)

def HF_solution(E, V, W, N=2):
    """
    Compute the Hartree-Fock solution for the ground state energy.

    Parameters:
    E (float): The energy.
    V (float): The potential.
    W (float): The interaction strength.
    N (int): The number of particles. Default is 2.

    Returns:
    float: The Hartree-Fock solution for the ground state energy.
    """
    v = V * (N - 1) / E
    if v < 1:
        return -N / 2 * E
    else:
        return -N / 2 * ((E**2 + (N - 1)**2 * V**2) / (2 * (N - 1) * V))


def plot_eVals_and_HF_sol(E, W_vals, V_range, filename):
    """
    Plot the eigenvalues and Hartree-Fock solution for a range of potential values.

    Parameters:
    E (float): The energy.
    W_vals (list of float): The interaction strengths to plot.
    V_range (np.array): The range of potential values.
    filename (str): The name of the file to save the plot to.

    Returns:
    None
    """
    fig, ax = plt.subplots()
    for W in W_vals:
        linestyle = '-' if W == 0 else '--'
        label_W = f'$W = {W}$'
        for mode, label in zip(range(3), [r'$\lambda_0$', r'$\lambda_1$', r'$\lambda_2$']):
            ax.plot(V_range, [eig(E, V, W, mode) for V in V_range], f'{linestyle}', label=f'{label}, {label_W}')
        if W == 0:  
            HF_sol = [HF_solution(E, V, W) for V in V_range]
            ax.plot(V_range, HF_sol, 'b:', label=r'HF solution, $W = 0$')
    ax.set_xlabel('V')
    ax.set_ylabel('Energy')
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, format='png')


E = 1.0
V = np.linspace(0, 2, 20)
plot_eVals_and_HF_sol(E, [0, -1.5], V, 'J1_V_W.png')


def find_min_eVals(E, W, V_range):
    """
    Compute the minimum eigenvalues for a range of potential values using the NumPyMinimumEigensolver.

    Parameters:
    E (float): The energy.
    W (float): The interaction strength.
    V_range (np.array): The range of potential values.

    Returns:
    np.array: The minimum eigenvalues for each potential value in V_range.
    """
    numpy_eigs = np.zeros(len(V_range))
    observable_template = [("IZ", E / 2), ("ZI", E / 2), ("XX", 0.5), ("YY", 0.5)]
    
    for i, V in enumerate(V_range):
        observable = SparsePauliOp.from_list([(op[0], op[1] * ((W - V) if op[0] in ["XX"] else (W + V))) for op in observable_template])
        result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(operator=observable)
        numpy_eigs[i] = result.eigenvalue.real
    return numpy_eigs


numpy_eigs = find_min_eVals(E, -1.5, np.linspace(0, 2, num=20))


#######################################################################################
"The Qiskit part is bellow"
######################################################################################
n_Vs = 20
Vs = np.linspace(0, 2, n_Vs)
e = 1


def Hamil(e, V, W):
    """
    Construct the Hamiltonian matrix.

    Parameters:
    e (float): The energy.
    V (float): The potential.
    W (float): The interaction strength.

    Returns:
    np.array: The Hamiltonian matrix.
    """

    return np.array([
        [-2*e,          0,       np.sqrt(6)*V, 0,       0],
        [0,            -e + 3*W, 0,            3*V,     0],
        [np.sqrt(6)*V,  0,       4*W,          0,       np.sqrt(6)*V],
        [0,             3*V,     0,            e + 3*W, 0],
        [0,             0,       np.sqrt(6)*V, 0,       2*e]
    ])


def HF_solution(E, V, W, N=2):
    """

    Compute the Hartree-Fock solution for the ground state energy.

    Parameters:
    E (float): The energy.
    V (float): The potential.
    W (float): The interaction strength.
    N (int): The number of particles. Default is 2.

    Returns:
    float: The Hartree-Fock solution for the ground state energy.
    """

    v = V*(N-1)/E
    return -N/2*E if v < 1 else -N/2*((E**2 + (N-1)**2*V**2)/(2*(N-1)*V))


def comp_eigsys(Vs, W):
    """
    Compute the eigenvalues and eigenvectors for a range of potential values.

    Parameters:
    Vs (np.array): The range of potential values.
    W (float): The interaction strength.

    Returns:
    tuple: The eigenvalues and eigenvectors for each potential value in Vs.
    """

    eVals = np.zeros((n_Vs, 5))
    eVecs = np.zeros((n_Vs, 25))
    for i, V in enumerate(Vs):
        H = Hamil(e, V, W)
        EigValues, EigVectors = np.linalg.eig(H)
        permute = EigValues.argsort()
        eVals[i,:] = EigValues[permute]
        eVecs[i,:] = EigVectors[:, permute].flatten()
    return eVals, eVecs


def plot_eigsys(Vs, eVals, W, label_suffix, ax, style='-'):
    """
    Plot the eigenvalues for a range of potential values.

    Parameters:
    Vs (np.array): The range of potential values.
    eVals (np.array): The eigenvalues for each potential value in Vs.
    W (float): The interaction strength.
    label_suffix (bool): Whether to add a suffix to the label.
    ax (matplotlib.axes.Axes): The axes to plot on.
    style (str): The line style. Default is '-'.

    Returns:
    None
    """

    for i in range(eVals.shape[1]):
        ax.plot(Vs, eVals[:,i], f'{style}', label=f'$\\epsilon_{i}, W={W}$' if label_suffix else None)


fig, ax = plt.subplots(figsize=(6,4))
for W, style in zip([0, -0.5], ['-', '--']):
    eVals, _ = comp_eigsys(Vs, W)
    plot_eigsys(Vs, eVals, W, label_suffix=True, ax=ax, style=style)

HF_sol = [HF_solution(e, V, 0, N=4) for V in Vs]
ax.plot(Vs, HF_sol, 'b:', label=r'HF solution, $W = 0$')
ax.set_xlabel(r'$V$')
ax.set_ylabel('energy')
ax.legend()
fig.tight_layout()
fig.savefig('J2_W05.png', format='png')


n_Vs = 10
Vs = np.linspace(0, 2, n_Vs)
E = e  

# Setup Quantum Instance
backend = Aer.get_backend('aer_simulator')
estim  = AerEstimator(run_options={"shots": 1024})


# Observable Creation Function
def observable(E, V, W):
    """
    Construct the observable operator.

    Parameters:
    E (float): The energy.
    V (float): The potential.
    W (float): The interaction strength.

    Returns:
    SparsePauliOp: The observable operator.
    """
    pauli_terms = [
        ("ZIII", E/2), ("IZII", E/2), ("IIZI", E/2), ("IIIZ", E/2),
        ("XXII", (W+V)/2), ("XIXI", (W+V)/2), ("XIIX", (W+V)/2),
        ("IXXI", (W+V)/2), ("IXIX", (W+V)/2), ("IIXX", (W+V)/2),
        ("YYII", (W-V)/2), ("YIYI", (W-V)/2), ("YIIY", (W-V)/2),
        ("IYYI", (W-V)/2), ("IYIY", (W-V)/2), ("IIYY", (W-V)/2)
    ]
    return SparsePauliOp.from_list(pauli_terms)

# Function to Run Qiskit Algorithms
def qiskit_algo(Vs, E, W):
    """
    Run the NumPyMinimumEigensolver and VQE algorithms for a range of potential values.

    Parameters:
    Vs (np.array): The range of potential values.
    E (float): The energy.
    W (float): The interaction strength.

    Returns:
    tuple: The eigenvalues computed by the NumPyMinimumEigensolver and VQE algorithms for each potential value in Vs.
    """
    numpy_eigs = []
    vqe_eigs = []
    
    for V in Vs:
        observable_var = observable(E, V, W)
        
        # NumPyMinimumEigensolver
        result_numpy = NumPyMinimumEigensolver().compute_minimum_eigenvalue(operator=observable_var)
        numpy_eigs.append(result_numpy.eigenvalue.real)
        
        # VQE
        ansatz = TwoLocal(4, rotation_blocks=["rx", "ry"], entanglement_blocks='cz', entanglement='full', reps=2)
        optimizer = SPSA(maxiter=2000)
        vqe = VQE(estim, ansatz, optimizer=optimizer)
        result_vqe = vqe.compute_minimum_eigenvalue(operator=observable_var)
        vqe_eigs.append(result_vqe.eigenvalue.real)
    
    return numpy_eigs, vqe_eigs


numpy_eigs, vqe_eigs = qiskit_algo(Vs, E, W=-0.5)


fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.plot(Vs, numpy_eigs, 'bo', label='Numpy min eigensolver')
ax2.plot(Vs, vqe_eigs, 'rx', label='Qiskit VQE')
ax2.set_xlabel(r'$V$')
ax2.set_ylabel('energy')
ax2.legend()
fig2.tight_layout()
fig2.savefig('J2_W05_VQE.png', format='png')
