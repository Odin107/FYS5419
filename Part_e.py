import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qiskit_aer import Aer
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Pauli, SparsePauliOp
from Part_c import ansatz

# Defining the quantum backends
state_simulator = Aer.get_backend('statevector_simulator') #can be removed
simulator = Aer.get_backend('aer_simulator')

# Define Pauli operators
I = Operator(Pauli('I'))
X = Operator(Pauli('X'))
Y = Operator(Pauli('Y'))
Z = Operator(Pauli('Z'))

# Constants
lmd_ = 0.4
eps_list = [0, 2.5, 6.5, 7]
H_x = 2.0
H_z = 3.0

# State vectors
s_0 = np.array([1, 0])
s_1 = np.array([0, 1])

# Hamiltonians
# H0 = np.diag(eps_list)  # Diagonal matrix with eps_list as diagonal elements
H0 = np.eye(4)
H0[0,0] = eps_list[0];H0[1,1] = eps_list[1];H0[2,2] = eps_list[2];H0[3,3] = eps_list[3]
pauli_x = np.array([[0, 1], [1, 0]])
pauli_z = np.array([[1, 0], [0, -1]])
H_I = H_x * np.kron(pauli_x, pauli_x) + H_z * np.kron(pauli_z, pauli_z)
H_tot = H0 + lmd_ * H_I

# Assuming H0 and H1 are defined elsewhere in your code.

n = 1000
lambdas = np.linspace(0,1,n)
Es = np.zeros((n,4))
C1s, C2s  = np.zeros_like(Es), np.zeros_like(Es)

for i, lmd_ in enumerate(lambdas):
    H_ = H0 + lmd_*H_I
    eig_val, eig_vec = np.linalg.eigh(H_)
    Es[i,:] = eig_val

fig, ax = plt.subplots()
ax.plot(lambdas, Es[:,0], label="E_0")
ax.plot(lambdas, Es[:,1], label="E_1")
ax.set(xlabel="lambda", ylabel="E")
ax.legend() 
plt.savefig('/home/odinjo/Project_1/QML_project_1/figures/figs_part_e/eigenvalues_plot.png')


# H_0 = SparsePauliOp.from_list([(Pauli(p), c) for p, c in zip(["IIZZ", "IZIZ", "IZZI", "ZIIZ", "ZIZI", "ZZII"], eps_list)])

# Interaction Hamiltonian terms
H_I_terms = [('XXII', H_x), ('YYII', H_x), ('ZZII', H_z)]
H_I = SparsePauliOp.from_list(H_I_terms)

# Combine the Hamiltonians
H_0 = np.matrix([[eps_list[0],0,0,0],[0,eps_list[1],0,0],[0,0,eps_list[2],0],[0,0,0,eps_list[3]]])
H_I = H_x * np.kron(X,X) + H_z* np.kron(Z,Z) 
H = H_0 + H_I
H = H0 + H_I

# Calculate eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eigh(H)

print(eigvals)
print(eigvecs)

# Define constants
c_1 = 1
c_2 = 1

# Create the Hamiltonian list more succinctly
H_list = [[c_1, [0], ['Z', 'Z']], [c_2, [1], ['X', 'X']]]


n_qubits= 2
qreg = QuantumRegister(n_qubits)
circuit = QuantumCircuit(qreg)
circuit.h(qreg[:2])
print('Before ansatz')
print(circuit.draw())
theta = np.random.randn(2)
n_qubits = 2
circuit = circuit.compose(ansatz(theta,n_qubits))
print('After ansatz')
print(circuit.draw())


def basis_change(h_i,n_qubits):
    qreg = QuantumRegister(n_qubits)
    circuit = QuantumCircuit(qreg)
    
    for qubit,operator in zip(h_i[1],h_i[2]):
        if operator == 'x':
            circuit.h(qreg[qubit])
    return(circuit)
n_qubits = 2
qreg = QuantumRegister(n_qubits)
circuit = QuantumCircuit(qreg)
theta = np.random.randn(n_qubits)
circuit = circuit.compose(ansatz(theta,n_qubits))
print('Ansatz circuit')
print(circuit.draw())
circuit = circuit.compose(basis_change(H[1],n_qubits))
print('After basis transformation:')
print(circuit.draw())

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

theta = np.random.randn(2)
e, circ = energy(theta)
circ.draw()



def compute_gradient(theta, epsilon=1e-2):
    gradient = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_plus = np.copy(theta)
        theta_minus = np.copy(theta)
        theta_plus[i] += epsilon  #np.pi/10   
        theta_minus[i] -= epsilon #np.pi/10  
        energy_plus = energy(theta_plus)
        energy_minus = energy(theta_minus)
        gradient[i] = ((energy_plus - energy_minus) / (2 * epsilon))
    return gradient

def gradient_descent(theta, learning_rate, num_iterations):
    min_theta = theta
    min_eval  = energy(theta)
    for iteration in range(num_iterations):
        energy = energy(theta)
        gradient = compute_gradient(theta)
        theta -= (learning_rate * gradient)
        """
        if iteration%10 == True:
            print("Iteration:", iteration, "Energy:", energy)
        """
        if energy < min_eval:
            min_theta = theta 
            min_eval  = energy
    return min_theta, min_eval

theta = np.array([0.1,0.1])
learning_rate = 0.08
num_iterations = 1000

arg_best, arg_val  = gradient_descent(theta, learning_rate, num_iterations)
print("The lowest energy is:",arg_val,"with theta values of", arg_best)






















