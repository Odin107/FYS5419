import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import qiskit as qk
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram, plot_bloch_vector, plot_state_city, plot_bloch_multivector, plot_state_qsphere
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.circuit import QuantumCircuit, Parameter, QuantumRegister, ClassicalRegister
from qiskit import transpile

# Defining the quantum backends
state_simulator = Aer.get_backend('statevector_simulator')
simulator = Aer.get_backend('aer_simulator')



# Task a
def apply_gates_and_measure(initial_state = '0', gate_type='X', measure=False, shots=1024):
    """
    Apply a specified quantum gate to an initial state and optionally measure the result.

    Parameters:
    initial_state (str): The initial state of the qubit, either '0' or '1'. Default is '0'.
    gate_type (str): The type of quantum gate to apply. Options are 'H', 'S', 'X', 'Y', and 'Z'. Default is 'X'.
    measure (bool): Whether to measure the state after applying the gate. Default is False.
    shots (int): The number of measurements to perform if measure is True. Default is 1024.

    Returns:
    np.array or dict: The statevector after applying the gate, or the measurement results if measure is True.
    """
    qc = QuantumCircuit(1, 1 if measure else 0)

    if initial_state == '1':
        qc.x(0) 
    
    if gate_type == 'H':
        qc.h(0)


    elif gate_type == 'S':
        qc.s(0)

    
    elif gate_type == 'X':
        qc.x(0)

    
    elif gate_type == 'Y':
        qc.y(0)

    
    elif gate_type == 'Z':
        qc.z(0)
    
    if measure:
        qc.measure(0, 0)
    
    backend = Aer.get_backend('statevector_simulator' if not measure else 'qasm_simulator')
    transpiled_circuit = qk.transpile(qc, backend)
    result = backend.run(transpiled_circuit, shots=shots).result()

    return result.get_statevector() if not measure else result.get_counts()

print("Statevector after applying X gate:", apply_gates_and_measure('0', 'X'))
print("Measurement results after applying H gate:", apply_gates_and_measure('H', measure=True))



def bell_state_and_gates(bell_state_type='Phi+', measure=False, shots=1024):
    """
    Prepare a specified Bell state and optionally measure the result.

    Parameters:
    bell_state_type (str): The type of Bell state to prepare. Options are 'Phi+', 'Phi-', 'Psi+', and 'Psi-'. Default is 'Phi+'.
    measure (bool): Whether to measure the state after preparing the Bell state. Default is False.
    shots (int): The number of measurements to perform if measure is True. Default is 1024.

    Returns:
    np.array or dict: The statevector of the Bell state, or the measurement results if measure is True.
    """
    qc = QuantumCircuit(2, 2 if measure else 0)

    if bell_state_type in ['Phi+', 'Phi-', 'Psi+', 'Psi-']:
        qc.h(0)  
        qc.cx(0, 1)  

    if bell_state_type == 'Phi-':
        qc.z(0) 
    elif bell_state_type == 'Psi+':
        qc.x(1)  
    elif bell_state_type == 'Psi-':
        qc.x(1)
        qc.z(0)
    if measure:
        qc.measure([0, 1], [0, 1])
    
    backend = Aer.get_backend('statevector_simulator' if not measure else 'qasm_simulator')
    transpiled_circuit = qk.transpile(qc, backend)
    result = backend.run(transpiled_circuit, shots=shots).result()
    
    return result.get_statevector() if not measure else result.get_counts()


print("Statevector after applying S gate to |1⟩ state:", apply_gates_and_measure('1', 'S'))
print("Measurement results from the Ψ- Bell state:", bell_state_and_gates('Psi-', measure=True))


h_results = apply_gates_and_measure('0', 'H', measure=True, shots=1024)

plot_histogram(h_results)
plt.title("Histogram of Measurement Results after Applying H Gate")
plt.ylabel('Probability')
plt.savefig('/home/odinjo/Project_1/QML_project_1/figures/figs_part_a/histogram_h_gate.png')

bell_results = bell_state_and_gates('Phi+', measure=True, shots=1024)

plot_histogram(bell_results)
plt.title("Histogram of Measurement Results from Bell State")
plt.ylabel('Probability')
plt.savefig('/home/odinjo/Project_1/QML_project_1/figures/figs_part_a/histogram_phi_plus_bell_state.png')


# Initializing a bell state 
qr_bell = QuantumRegister(2, 'qreg')
cr_bell = ClassicalRegister(2, 'creg')
qc_bell = QuantumCircuit(qr_bell, cr_bell)


# Apply gates to the bell state
qc_bell.h(0)
qc_bell.cx(0, 1)


state_bell = Statevector(qc_bell)
plot_state_qsphere(state_bell, filename='/home/odinjo/Project_1/QML_project_1/figures/figs_part_a/q_sphere_bell.png')

# Measure all the bell states 
qc_bell.measure(0,0)
qc_bell.measure(1,1)

# Visualizing the circuit
qc_bell.draw(output='mpl', style='default', filename='/home/odinjo/Project_1/QML_project_1/figures/figs_part_a/quantum_circuit_bell.png')


jobb_bell_sim = simulator.run(qc_bell, shots=1024)
bell_hist = jobb_bell_sim.result().get_counts()
plot_histogram(bell_hist).savefig('/home/odinjo/Project_1/QML_project_1/figures/figs_part_a/histogram_bell.png')














