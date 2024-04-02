import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qiskit_aer import Aer
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.circuit import QuantumCircuit, Parameter, QuantumRegister, ClassicalRegister
from qiskit import transpile


# Defining the quantum backends
state_simulator = Aer.get_backend('statevector_simulator')
simulator = Aer.get_backend('aer_simulator')

# Define Pauli operators
I = Operator(Pauli('I'))
X = Operator(Pauli('X'))
Y = Operator(Pauli('Y'))
Z = Operator(Pauli('Z'))


v =1
I = np.eye(2)
H_4 = -(np.kron(Z,I) + np.kron(I, Z)) - np.sqrt(6)/2 * v *(np.kron(X, I) + np.kron(I, X) + np.kron(X, Z) - np.kron(Z, X))

def H4(v):
    reg = -(np.kron(Z,I) + np.kron(I, Z))
    ireg = - np.sqrt(6)/2 * v *(np.kron(X,I)+np.kron(I,X) +np.kron(X,Z)-np.kron(Z,X))
    ret = reg + ireg
    return ret
                                                                       
                                                                       
H_4_o = - Z - 3*v*X


eigval_even, eigvec_even = np.linalg.eig(H_4)
eigval_odd, eigvec_odd = np.linalg.eig(H_4_o)

print(eigval_even)
print(eigval_odd)

H_op = Operator(H_4)

theta = np.random.randn(2)
v_list = np.linspace(-2,2,100)
eigval_list = []
for i in range(len(v_list)):
    eig_temp, trash = np.linalg.eig(H4(v_list[i]))
    eigval_list.append(min(eig_temp))
    #print(v_list[i])
    #print(eig_temp)
    #print()
    
plt.plot(v_list, eigval_list)
plt.grid()
plt.savefig('/home/odinjo/Project_1/QML_project_1/figures/figs_part_f/eigval_vs_v.png')


def lipkin_func(v):
    h1_lip = [-1, [0,1], ['z']]
    h2_lip = [-np.sqrt(6)/2*v, [0], ['x']]
    h3_lip = [-np.sqrt(6)/2*v, [1], ['x']]
    h4_lip = [-np.sqrt(6)/2*v, [0,1], ['xz']]
    h5_lip = [np.sqrt(6)/2*v, [0,1], ['zx']]
    H4 = [h1_lip, h2_lip, h3_lip, h4_lip, h5_lip]
    return H4

def ansatz(theta, n_qubits):
    qreg = QuantumRegister(n_qubits)
    circuit = QuantumCircuit(qreg)
    param_list =[]
    for j in range(len(theta)):
        par = Parameter('theta %s' %j)
        param_list.append(par)
    circuit.ry(param_list[0],qreg[0])
    circuit.cry(param_list[1],qreg[0],qreg[1])
    dictus ={}
    for j in range(len(theta)):
        dictus[param_list[j]] = theta[j]
    circuit = circuit.assign_parameters(dictus)
    circuit = transpile(circuit, backend = simulator)
    return(circuit)

def basis_change(h_i , n_qubits):
    qreg = QuantumRegister(n_qubits)
    bas_circ = QuantumCircuit(qreg)
    bas_circ = bas_circ.compose(ansatz(theta, n_qubits))

    for qubit,operator in zip(h_i[1],h_i[2]):
        if operator == 'x':
            if qubit == 1:
                bas_circ.h(qreg[qubit])
            if qubit == 0:
                bas_circ.h(qreg[qubit])
        if operator == 'xz':
            bas_circ.h(qreg[qubit])
        if operator == 'zx':
            bas_circ.h(qreg[qubit+1])
    return(bas_circ)

def get_energy(theta,v, n_qubits=2):
    H_4_lip = lipkin_func(v)
    qreg = QuantumRegister(n_qubits)
    circuit = QuantumCircuit(qreg)
    circuit_list = []
    for idx,h_i in enumerate(H_4_lip):
        basis_change_circuit = basis_change(h_i,n_qubits)
        new_circuit = circuit.compose(basis_change_circuit)
        creg = ClassicalRegister(len(h_i[1]))
        new_circuit.add_register(creg)
        new_circuit.measure(qreg[h_i[1]],creg)
        circuit_list.append(new_circuit)
    shots = 1000
    job = simulator.run(circuit_list,shots=shots)
    E = np.zeros(len(circuit_list))
    for i in range(len(circuit_list)):
        result = job.result()
        counts = result.get_counts(i)
        for key,value in counts.items():
            e = 1
            for bit in key:
                if bit == '0':
                    e *= 1
                if bit == '1':
                    e *= -1
            E[i] += e*value
        E[i] *= H_4_lip[i][0]
    E /= shots

    #print(E)
    #print(np.mean(E))
    #print(np.std(E))
    return(np.sum(E))

get_energy([0,1], 10)

def compute_gradient(theta,v, epsilon=1e-2):
    gradient = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_plus = np.copy(theta)
        theta_minus = np.copy(theta)
        theta_plus[i] += epsilon  #np.pi/10   
        theta_minus[i] -= epsilon #np.pi/10  
        energy_plus = get_energy(theta_plus,v)
        energy_minus = get_energy(theta_minus,v)
        gradient[i] = ((energy_plus - energy_minus) / (2 * epsilon))
    return gradient

def gradient_descent(theta, learning_rate, num_iterations, v):
    min_theta = theta
    min_eval  = get_energy(theta,v)
    for iteration in range(num_iterations):
        energy = get_energy(theta,v)
        gradient = compute_gradient(theta,v)
        theta -= (learning_rate * gradient)
        print(energy)
        if energy < min_eval:
            min_theta = theta 
            min_eval  = energy
    return min_theta, min_eval

theta = np.random.randn(2)
learning_rate = 0.4
num_iterations = 100
v = 1
arg_best, arg_val  = gradient_descent(theta, learning_rate, num_iterations,v)
print("The lowest energy is:",arg_val,"with theta values of", arg_best)

epochs = 50
v = np.linspace(0,1,100)
min_eigval_list = []
#eps = 3
theta = np.random.randn(2)
learning_rate = 0.08

for i in range(len(v)):
    #a = lip_ham(v[i])
    arg_best, arg_val  = gradient_descent(theta, learning_rate, epochs, v[i])
    min_eigval_list.append(arg_val)
    print(v[i])
    print(arg_val)
    print()

plt.plot(v, min_eigval_list)
plt.savefig('/home/odinjo/Project_1/QML_project_1/figures/figs_part_f/lowest_eigval_vs_v.png')