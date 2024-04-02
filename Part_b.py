import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#Task b 

#The pauli matrices, using numpy arrays as they are faster qiskit's Pauli class
sigma_x = np.array([[0, 1], [1, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
I = np.eye(2)

#defining const
E1 = 0
E2 = 4
V11 = 3
V22 = -3
V12 = 0.2
V21 = 0.2
lambda_val = 1

E = (E1 + E2) / 2
Omega = (E1 - E2) / 2
c = (V11 + V22) / 2
omega_z = (V11 - V22) / 2
omega_x = V12


H0 = E * I + Omega * sigma_z
HI = c * I + omega_z * sigma_z + omega_x * sigma_x
H = H0 + (lambda_val * HI)


# Calculating and printing eigenvectors at critical lambda values
def eigvec_crit_lamb():
    critical_lambdas = [0, 2/3, 1]
    for lambda_val in critical_lambdas:
        H = H0 + lambda_val * HI
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        print(f"Eigenvectors at λ={lambda_val}:")
        for i in range(len(eigenvalues)):
            print(f"Eigenvalue {eigenvalues[i]}: Eigenvector {eigenvectors[:, i]}")
        print("\n")

eigvec_crit_lamb()


#solving the eigenvalue problem
lambdas = np.linspace(0, 1, 100)
eigenvalues = np.zeros((2, len(lambdas)))

for i, lambda_val in enumerate(lambdas):
    H = H0 + lambda_val * HI
    vals, _ = np.linalg.eigh(H)
    eigenvalues[:, i] = vals

plt.plot(lambdas, eigenvalues[0, :], label='Eigenvalue 1')
plt.plot(lambdas, eigenvalues[1, :], label='Eigenvalue 2')
plt.xlabel('Interaction strength (λ)')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues as Function of Interaction Strength')
plt.legend()
plt.grid(True)
plt.savefig('/home/odinjo/Project_1/QML_project_1/figures/figs_part_b/eigenvalues_vs_lambda.png')
print("Plot saved as 'eigenvalues_vs_lambda.png'.")


# Critical lambdas
critical_lambdas = [0, 2/3, 1]
labels = ['λ=0 (Non-interacting)', 'λ=2/3 (Transition point)', 'λ=1 (Fully interacting)']

eigenvector_components_critical = np.zeros((len(critical_lambdas), 2, 2))  # 3 lambda values, 2 components, 2 eigenvectors

for i, lambda_val in enumerate(critical_lambdas):
    H = H0 + lambda_val * HI
    vals, vecs = np.linalg.eigh(H)
    eigenvector_components_critical[i, :, :] = vecs

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for i in range(2):  # Loop over each eigenvector
    ax = axes[i]
    ax.bar(np.arange(len(critical_lambdas))-0.1, np.abs(eigenvector_components_critical[:, 0, i]), width=0.2, label='Component $|0\\rangle$')
    ax.bar(np.arange(len(critical_lambdas))+0.1, np.abs(eigenvector_components_critical[:, 1, i]), width=0.2, label='Component $|1\\rangle$')
    ax.set_xticks(np.arange(len(critical_lambdas)))
    ax.set_xticklabels(labels)
    ax.set_title(f'Eigenvector {i+1} Components at Critical λ')
    ax.set_ylabel('Component Magnitude' if i == 0 else '')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('/home/odinjo/Project_1/QML_project_1/figures/figs_part_b/eigenvalues_vs_critical_lambda.png')
print("Plot saved as 'eigenvalues_vs_critical_lambda.png'.")