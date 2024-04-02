import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import logm


X, Y, Z, I = [np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.array([[1,0],[0,-1]]), np.eye(2)]

# System parameters
dim, E, Hx, Hz = 4, [0.0, 2.5, 6.5, 7.0], 2.0, 3.0
H0 = np.diag(E)
HI = Hx * np.kron(X, X) + Hz * np.kron(Z, Z)

# Projection operators for a single qubit system
P0, P1 = [np.array([[1,0],[0,0]]), np.array([[0,0],[0,1]])]
Proj = [np.kron(P, I) for P in [P0, P1]]

def density(state):
    """
    Compute the density matrix of a given state.

    Parameters:
    state (np.array): The state vector.

    Returns:
    np.array: The density matrix of the state.
    """
    return np.outer(state, np.conj(state))

def log2M(a):
    """
    Compute the base-2 logarithm of a matrix.

    Parameters:
    a (np.array): The input matrix.

    Returns:
    np.array: The base-2 logarithm of the matrix.
    """
    return logm(a) / np.log(2.0)

def trace_out(rho, index):
    """
    Trace out a subsystem from a density matrix.

    Parameters:
    rho (np.array): The density matrix.
    index (int): The index of the subsystem to trace out.

    Returns:
    np.array: The resulting density matrix after tracing out the subsystem.
    """
    nbits = int(np.log2(rho.shape[0]))
    assert index < nbits, 'Invalid index: Index exceeds the number of qubits.'
    
    zero, one = np.array([1, 0]), np.array([0, 1])
    
    # Projection matrices initialization
    p0, p1 = [sum(np.kron(np.eye(2**index), np.kron(basis, np.eye(2**(nbits-index-1)))) for basis in (state,)) @ rho 
              for state in ((zero,), (one,))]
    
    return p0 @ p0.T + p1 @ p1.T


def entropy(evects, state):
    """
    Compute the entropy of a state.

    Parameters:
    evects (np.array): The eigenvectors of the system.
    state (int): The index of the state.

    Returns:
    float: The entropy of the state.
    """
    rho = density(evects[:, state])
    red_rho = trace_out(rho, state)
    return -np.trace(red_rho @ log2M(red_rho))

num_lambdas_ = 50
lambdas_ = np.linspace(0, 1, num=num_lambdas_)
eVals, entropies = np.zeros((num_lambdas_, 4)), np.zeros(num_lambdas_)

for i, lambd in enumerate(lambdas_):
    eVals[i], eVecs = np.linalg.eigh(H0 + lambd*HI)
    entropies[i] = entropy(eVecs, 0)


fig, ax = plt.subplots()
for i in range(4):
    ax.plot(lambdas_, eVals[:, i], label=f'$\epsilon_{i}$')
ax.set(xlabel=r'$\lambda$', ylabel='energy')
ax.legend()
fig.tight_layout()
plt.grid(True)
fig.savefig('eigval.png', format='png')

fig2, ax2 = plt.subplots()
ax2.plot(lambdas_, entropies, label = 'entropy')
ax2.set(xlabel=r'$\lambda$', ylabel='entropy')
ax2.legend()
fig2.tight_layout()
plt.grid(True)
fig2.savefig('entropies.png', format='png')




