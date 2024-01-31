import pennylane as qml
from pennylane import numpy as np, AdamOptimizer
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import load_digits
import scipy

# ---------------------------------------【user input】---------------------------------
nqubits = 6 # Number of data qubits, each data point in the load_digits dataset is represented with 8x8 dimensions and rolled into a 64*1 dimensional vector
row = 8 # row of one digit
col = 8 # column of one digit

mqubits = 2
dataNum = 2 ** mqubits

mnlayers = 36  # data and address layers
mnqubits = mqubits + nqubits

max_iterations = 2000
error = 1e-6

opt = AdamOptimizer(stepsize=0.2, beta1=0.9, beta2=0.999)

print('** User input *: ', 'mqubits =', mqubits, ' nqubits =', nqubits)
print('Quantum circuit: ', 'mnlayers =', mnlayers)
print()
# ---------------------------------------【End user input】---------------------------------

# ---------------------------------------【Stage 1: Data Orthogonalization】---------------------------------
### Loading M data points, where M = dataNum = 2**mqubits
digits = load_digits()
data = []
for i in range(dataNum):
    temp = digits.data[i]
    temp = temp / np.linalg.norm(temp)
    data.append(temp)

### Compute S, sqrt_S, and inv_sqrt_S, where S_ij = <x_i|x_j>
S = np.dot(data, np.array(data).T)   # compute S
def sqrt_matrix(S):
    # Calculating the square root of a matrix through eigenvalue decomposition
    w, v = np.linalg.eigh(S)
    w_sqrt = np.sqrt(w.clip(min=0))  # calculates the square root of each element in the array w.
    sqrt_S = v.dot(np.diag(w_sqrt)).dot(v.T)
    return sqrt_S

### Computing the square root of matrix S
sqrt_S = sqrt_matrix(S)  # sqrt_S = sqrtm(S) # # Note: sqrt_S may be complex when using sqrtm(S)

### Calculating the inverse square root of matrix S
inv_sqrt_S = np.linalg.inv(sqrt_S)

### Transforming the data using the inverse square root of S
dataTilde = np.matmul(inv_sqrt_S, data)

# ---------------------------------------【Stage 2: QRAM circuit for orthogonalized data】---------------------------------

### Convert data and dataTilde to quantum states with normalization factor 1
psi_data = []
for i in range(dataNum):
    psi_data = np.concatenate((psi_data, digits.data[i]))

psi_data = psi_data.reshape(dataNum * row * col)
psi_data = psi_data / np.linalg.norm(psi_data)

psi_dataTilde = np.array(dataTilde)
psi_dataTilde = psi_dataTilde.reshape(dataNum * row * col)
psi_dataTilde = psi_dataTilde / np.linalg.norm(psi_dataTilde)


###  Train for QRAM circuit

dev_mn = qml.device('default.qubit', wires=mnqubits)
dev_n = qml.device('default.qubit', wires=nqubits)
dev_m = qml.device('default.qubit', wires=mqubits)

def init_params():
    np.random.seed(1)
    params = np.random.randn(mnlayers, nqubits)
    return params

def nCNOT2(start, end):
    """
    :param start: the number of start qubit
    :param end: the number of end qubit
    """
    for i in range(start, end, 2):
        qml.CNOT(wires=[i, i+1])
    for i in range(start+1, end, 2):
        qml.CNOT(wires=[i, i+1])
    qml.CNOT(wires=[end, start])

### Train QRAM PQCs for orthogonal data
def train_QRAM_Orthogonal():

    def ansatz(params, wires):
        for i in range(mqubits):  # Hadamard on address qubits
            qml.Hadamard(wires=i)
        for i in range(mqubits):  # CNOT on qubits [addrqubits + dataqubits]
            qml.CNOT(wires=[i, i+mqubits])
        # fullAnsz on qubits [mqubits, mqubits+nqubits-1], i.e. U_x_Tilde(theta)
        for i in range(mnlayers):
            for j in range(nqubits):
                qml.RY(params[i, j], wires=j+mqubits)
            nCNOT2(mqubits, mqubits+nqubits-1)

    @qml.qnode(dev_mn)
    def final_circuit(params, wires=range(mnqubits)):
        ansatz(params, wires)
        return qml.state()

    def cost_mse(params): # mean_squared_error : mse = 1/n * sum((yi - xi) ** 2)
        psi_out = np.real(final_circuit(params, nqubits))
        loss = np.mean((psi_dataTilde - psi_out) ** 2)
        return loss

    def output_psi(params, wires=nqubits):
        result = np.real(final_circuit(params, wires=mnqubits))
        fidelity = np.dot(psi_dataTilde, result)
        return result, fidelity

    params = init_params()
    it = 0  # the iteration number, starting from 0
    _, fidelity = output_psi(params)

    ## Iteratively optimize the circuit parameters until fidelity threshold is met or max_iterations is reached
    while(fidelity < 0.999) and it < max_iterations:
        params = opt.step(lambda v: cost_mse(v), params)  # Perform optimization step

        ## Print progress every 100 iterations
        if (it % 100 == 0):
            loss = cost_mse(params)  # Calculate the current loss
            _, fidelity = output_psi(params)
            res = [it + 1, loss, fidelity]
            print(
                "Iteration: {:2d} | Loss: {:3f} ｜ fidelity: {:3f} ".format(
                    *res
                )
            )
        it += 1

    # Calculate the final quantum state and fidelity after optimization
    result, fidelity = output_psi(params)
    return params, result, fidelity

""" Train the orthogonal QRAM circuit and obtain parameters, quantum state, and fidelity score """
params_U1, psi_Orth_QRAM, fidelity_dataTilde = train_QRAM_Orthogonal()

""" Save the trained parameters, quantum state, and fidelity score into files """
with open("./pkl/params_U1.pkl", "wb") as f:
    pickle.dump(params_U1, f)
with open("./pkl/psi_Orth_QRAM.pkl", "wb") as f:
    pickle.dump(psi_Orth_QRAM, f)
with open("./pkl/fidelity_dataTilde.pkl", "wb") as f:
    pickle.dump(fidelity_dataTilde, f)

"""  Load trained QRAM parameters, quantum state, and fidelity score from saved files """
# with open("./pkl/params_U1.pkl", "rb") as f:
#     params_U1 = pickle.load(f)
# with open("./pkl/psi_Orth_QRAM.pkl", "rb") as f:
#     psi_Orth_QRAM = pickle.load(f)
# with open("./pkl/fidelity_dataTilde.pkl", "rb") as f:
#     fidelity_dataTilde = pickle.load(f)

# ---------------------------------------【Stage 3: QRAM circuit for the reverse transformation】---------------------------------

time = 0.68  # better

def QRAM_Final(out_params, A, time):

    U1 = scipy.linalg.expm(1j * A * time)
    U2 = scipy.linalg.expm(-1j * A * time)

    dev = qml.device('default.qubit', wires=1+mqubits+nqubits)

    def controlled_Unitary():
        qml.Hadamard(0)
        qml.PauliX(0)
        qml.ControlledQubitUnitary(U1, control_wires=[0], wires=[1,2]) # Notice: The number of wires equals mqubits
        qml.PauliX(0)
        qml.ControlledQubitUnitary(U2, control_wires=[0], wires=[1,2])
        qml.Hadamard(0)  # Transform |+⟩ -> |0⟩ and transform |-⟩ -> |1⟩,
                         # and our goal is to obtain |-⟩, which means the first qubit needs to be |1⟩ in this case.

    @qml.qnode(dev)
    def circuit(out_params):  ### H^mqubits + CNOT(mqubits->nqubits) + fullAnsz ==> QRAM
        for i in range(mqubits):  # Apply Hadamard gate on address qubits
            qml.Hadamard(wires=i+1)
        for i in range(mqubits):  # Apply CNOT gate on qubits [addrqubits + dataqubits]
            qml.CNOT(wires=[i+1, i+1+mqubits])
        # fullAnsz on qubits [mqubits+1, mqubits+nqubits]
        for i in range(mnlayers):
            for j in range(nqubits):
                qml.RY(out_params[i, j], wires=j+1+mqubits)
            nCNOT2(mqubits+1, mnqubits)
        controlled_Unitary()
        return qml.probs(wires=range(0, 1+mnqubits))

    out = circuit(out_params)           # 1 + mnqubits  # print(out.shape)  # (512,)
    prob_1 = np.sum(out[256:512])       # Probability of measuring state |1⟩
    result = np.sqrt(out[256:512]) / np.sqrt(prob_1)  # Notice: normalization

    # qml.draw_mpl(circuit, expansion_strategy="device", fontsize='xx-large')(out_params)
    # plt.show()
    return result

### Run the final QRAM circuit and obtain the final quantum state as psi_Final_QRAM
psi_Final_QRAM = QRAM_Final(params_U1, sqrt_S, time)

# ---------------------------------------【Plot the quantum state】---------------------------------
def plotResult():
    def plotOriginaldata():
        data = psi_data.reshape(dataNum * row, col)

        plt.subplot(1,4,1)
        plt.imshow(data, cmap='gray')
        plt.title('Original_data')

    def plotOrthogonalData():
        data_Orth = psi_dataTilde.reshape(dataNum * row, col)

        plt.subplot(1, 4, 2)
        plt.imshow(data_Orth, cmap='gray')
        plt.title('data_Orth_byS')

    def plotQRAMOrthogonal():
        ##  Plot the quantum state after "Stage 2: QRAM circuit for orthogonalized data"
        Orth_QRAM = psi_Orth_QRAM.reshape(dataNum * row, col)

        plt.subplot(1, 4, 3)
        plt.imshow(Orth_QRAM, cmap='gray')
        plt.title('Orth_QRAM')

    def plotFinalQRAM():
        ##  Plot the final quantum state after "Stage 3: QRAM circuit for the reverse transformation"
        Final_QRAM = psi_Final_QRAM.reshape(dataNum * row, col)

        plt.subplot(1, 4, 4)
        plt.imshow(Final_QRAM, cmap='gray')
        plt.title('Final_QRAM')

    plotOriginaldata()
    plotOrthogonalData()
    plotQRAMOrthogonal()
    plotFinalQRAM()
    plt.show()

plotResult()

fidelity_orth = np.dot(psi_dataTilde, psi_Orth_QRAM)
fidelity_final = np.dot(psi_data, psi_Final_QRAM)
print('fidelity_orth = ', fidelity_orth)
print('fidelity_final = ', fidelity_final)
print('End.\n')
