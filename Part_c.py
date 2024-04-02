from qiskit_aer import Aer
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info.operators import Operator, Pauli
import numpy as np
from scipy.optimize import minimize


# Define the quantum registers and classical registers
qr_vqe = QuantumRegister(2, 'qreg')
cr_vqe = ClassicalRegister(2, 'creg')

# Create a quantum circuit
qc_vqe = QuantumCircuit(qr_vqe, cr_vqe)

# Define the quantum backend
simulator = Aer.get_backend('aer_simulator')

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

# Define the Hamiltonian terms
h11 = [c + eps, [0,1], [I]]
h12 = [sgm, [0], [Z]]
h21 = [omg_z, [0], [Z]]
h22 = [omg_x, [0], [X]]
H = [h11, h12, h21, h22]


# Define the ansatz circuit
def ansatz(theta, n_qubits):
    """
    Construct a quantum circuit representing the ansatz.

    Args:
        theta (list): A list of parameters for the ansatz.
        n_qubits (int): The number of qubits in the circuit.

    Returns:
        QuantumCircuit: The constructed quantum circuit.
    """
    qr = QuantumRegister(n_qubits)
    qc = QuantumCircuit(qr)
    parameters = ParameterVector('theta', length=len(theta))

    for i in range(n_qubits):
        qc.ry(parameters[i], qr[i])
    for i in range(n_qubits - 1):
        qc.cx(qr[i], qr[i+1])
    
    param_dict = {parameters[i]: theta_val for i, theta_val in enumerate(theta)}
    qc = qc.assign_parameters(param_dict)
    
    return qc

# plot of the ansatz circuit.
cc = ansatz([0, 1], 2)
cc.draw(output='mpl', style='default', filename='/home/odinjo/Project_1/QML_project_1/figures/figs_part_c/quantum_circuit_ansatz.pdf')


# Define a function to perform basis change
def basis_change(h_i, n_qubits):
    qr = QuantumRegister(n_qubits)
    qc = QuantumCircuit(qr)
    
    # Loop over each qubit and corresponding operator
    for qubit, operator in zip(h_i[1], h_i[2]):
        # Apply Hadamard gate if the operator is 'X'
        if np.any(operator == 'X'):
            qc.h(qr[qubit])
    
    return qc

# Define a function to calculate the energy
def energy(theta):
    """
    Calculate the expectation value of a Hamiltonian H for a given ansatz and set of parameters.

    This function constructs a quantum circuit for each term in the Hamiltonian using the provided ansatz and parameters.
    It then measures each circuit and calculates the expectation value of the corresponding Hamiltonian term.
    The expectation values are then summed to give the total energy.

    Parameters:
    theta (numpy.ndarray): An array of parameters for the ansatz.

    Returns:
    float: The expectation value of the Hamiltonian.

    Note:
    The Hamiltonian H and the ansatz function must be defined in the same scope as this function.
    The ansatz function should take two arguments: a list or array of parameters, and the number of qubits.
    """
    n_qubits = 2
    qreg = QuantumRegister(n_qubits)
    qc_base = QuantumCircuit(qreg).compose(ansatz(theta, n_qubits))
    qc_list = []
    
    for h_i in H:

        basis_change_circuit = basis_change(h_i, n_qubits)
        qc = qc_base.compose(basis_change_circuit)
        
        creg = ClassicalRegister(len(h_i[1]))
        qc.add_register(creg)
        qc.measure(qreg[h_i[1]], creg)
        qc_list.append(qc)
    
    shots = 1000
    results = simulator.run(qc_list, shots=shots).result()
    
    # Calculate the expectation value of the Hamiltonian
    E = np.array([sum((-1)**int(bit) * count for bit, count in results.get_counts(i).items()) * h_i[0] / shots for i, h_i in enumerate(H)])
    
    return E.sum()


# Calculate and print the energy
theta = np.random.randn(2)
print("This is the energy:", energy(theta))

# Optimize the energy using the Powell method
res = minimize(energy, theta, method='Powell', tol=1e-12)

# Calculate and print the optimized energy
print("Optimized enenrgy:", energy(res.x))


def compute_gradient(theta, epsilon=1e-2):
    gradient = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += epsilon
        theta_minus[i] -= epsilon
        gradient[i] = (energy(theta_plus) - energy(theta_minus)) / (2 * epsilon)
    return gradient

def gradient_descent(theta, learning_rate, num_iterations):
    min_theta = np.copy(theta)
    min_eval = energy(theta)  # Use your energy function to initialize min_eval
    for iteration in range(num_iterations):
        gradient = compute_gradient(theta)
        theta -= learning_rate * gradient
        current_energy = energy(theta)  # Use your energy function to calculate current energy
        if current_energy < min_eval:
            min_theta = np.copy(theta)
            min_eval = current_energy
        # Optional: Print progress every 50 iterations
        if iteration % 50 == 0:
            print(f"Iteration: {iteration}, Energy: {current_energy}")
        
    return min_theta, min_eval

theta = np.array([0.1, 0.1])
learning_rate = 0.12
num_iterations = 1000

arg_best, arg_val = gradient_descent(theta, learning_rate, num_iterations)
print(f"The lowest energy is: {arg_val}, with theta values of {arg_best}")



