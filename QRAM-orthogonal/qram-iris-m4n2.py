import pennylane as qml
from pennylane import numpy as np, AdamOptimizer
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import load_iris
import scipy
# ---------------------------------------【user input】---------------------------------
mqubits = 4  # Number of address qubits (m > n)
nqubits = 2  # Number of data qubits, the load_iris features are represented as 4*1 dimensions

mnqubits = mqubits + nqubits
realDataNum = 2 ** mqubits
realDataDim = 2 ** nqubits

groupAddr_qubits = nqubits
groupData_qubits = mqubits
groupNum = 2 ** groupAddr_qubits  # groupNum <= groupDim
groupDim = 2 ** groupData_qubits

print(f"groupAddr_qubits={groupAddr_qubits}, groupData_qubits={groupData_qubits}")
dataNumPerGroup = groupDim // groupNum

mnlayers = 12  # data and address layers
max_iterations = 2000
error = 1e-6

opt = AdamOptimizer(stepsize=0.2, beta1=0.9, beta2=0.999)

print('** User input *: ', 'mqubits =', mqubits, ' nqubits =', nqubits)
print('Quantum circuit: ', 'mnlayers =', mnlayers)
print()
# ---------------------------------------【End user input】---------------------------------

# ---------------------------------------【Stage 1: Data Orthogonalization】---------------------------------
### Loading M data points, where M = dataNum = 2**mqubits
iris = load_iris()

def get_original_data():
    psi_data = []
    for i in range(realDataNum):
        psi_data = np.concatenate((psi_data, iris.data[i]))
    psi_data = psi_data.reshape(realDataNum * realDataDim)
    psi_data = psi_data / np.linalg.norm(psi_data)
    return psi_data

psi_original_data = get_original_data()

### In data_grouping, each row contains 4 data points grouped together.
def data_grouping():
    data = psi_original_data.reshape(groupNum, groupDim) # groupNum = 4, groupDim = 16
    for k in range(groupNum):
        data[k] = data[k]/np.linalg.norm(data[k])
    return data
data = data_grouping()


### Compute S, sqrt_S, and inv_sqrt_S, where S_ij = <x_i|x_j>
S = np.dot(data, np.array(data).T)   # compute S

def sqrt_matrix(S):
    # Calculating the square root of a matrix through eigenvalue decomposition
    w, v = np.linalg.eigh(S)
    w_sqrt = np.sqrt(w.clip(min=0))
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
for i in range(groupNum):
    psi_data = np.concatenate((psi_data, data[i]))

psi_data = psi_data.reshape(groupNum * groupDim)
psi_data = psi_data / np.linalg.norm(psi_data)

psi_dataTilde = np.array(dataTilde)
psi_dataTilde = psi_dataTilde.reshape(groupNum * groupDim)
psi_dataTilde = psi_dataTilde / np.linalg.norm(psi_dataTilde)


###  Train for QRAM circuit

dev_mn = qml.device('default.qubit', wires=mnqubits)
dev_n = qml.device('default.qubit', wires=nqubits)
dev_m = qml.device('default.qubit', wires=mqubits)

def init_params():
    np.random.seed(2)
    params = np.random.randn(mnlayers, groupData_qubits)
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
        for i in range(groupAddr_qubits):  # Hadamard on groupAddress qubits
            qml.Hadamard(wires=i)
        for i in range(groupAddr_qubits):  # CNOT on qubits [addrqubits + dataqubits]
            qml.CNOT(wires=[i, i+groupAddr_qubits])
        # fullAnsz on qubits [mqubits, mqubits+nqubits-1], i.e. U_x_Tilde(theta)
        for i in range(mnlayers):
            for j in range(groupData_qubits):
                qml.RY(params[i, j], wires=j + groupAddr_qubits)
            nCNOT2(groupAddr_qubits, mnqubits-1)

    @qml.qnode(dev_mn)
    def final_circuit(params, wires=range(mnqubits)):
        ansatz(params, wires)
        return qml.state()

    def cost_mse(params): # mean_squared_error : mse = 1/n * sum((yi - xi) ** 2)
        psi_out = np.real(final_circuit(params, mnqubits))
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
with open("./pkl/params_U1_iris.pkl", "wb") as f:
    pickle.dump(params_U1, f)
with open("./pkl/psi_Orth_QRAM_iris.pkl", "wb") as f:
    pickle.dump(psi_Orth_QRAM, f)
with open("./pkl/fidelity_dataTilde_iris.pkl", "wb") as f:
    pickle.dump(fidelity_dataTilde, f)

"""  Load trained QRAM parameters, quantum state, and fidelity score from saved files """
# with open("./pkl/params_U1_iris.pkl", "rb") as f:
#     params_U1 = pickle.load(f)
# with open("./pkl/psi_Orth_QRAM_iris.pkl", "rb") as f:
#     psi_Orth_QRAM = pickle.load(f)
# with open("./pkl/fidelity_dataTilde_iris.pkl", "rb") as f:
#     fidelity_dataTilde = pickle.load(f)

# ---------------------------------------【Stage 3: QRAM circuit for the reverse transformation】---------------------------------

time = 0.8

def QRAM_Final(out_params, A, time):

    U1 = scipy.linalg.expm(1j * A * time)
    U2 = scipy.linalg.expm(-1j * A * time)

    dev = qml.device('default.qubit', wires=1+mnqubits)

    def controlled_Unitary():
        qml.Hadamard(0)
        qml.PauliX(0)
        qml.ControlledQubitUnitary(U1, control_wires=[0], wires=range(1,groupAddr_qubits+1)) # Notice: The number of wires equals groupAddr_qubits
        qml.PauliX(0)
        qml.ControlledQubitUnitary(U2, control_wires=[0], wires=range(1,groupAddr_qubits+1))
        qml.Hadamard(0)  # Transform |+⟩ -> |0⟩ and transform |-⟩ -> |1⟩,
                         # and our goal is to obtain |-⟩, which means the first qubit needs to be |1⟩ in this case.
    @qml.qnode(dev)
    def circuit(out_params):
        for i in range(groupAddr_qubits):  ### H^mqubits + CNOT(mqubits->nqubits) + fullAnsz ==> QRAM
            qml.Hadamard(wires=i+1)        # Apply Hadamard gate on groupAddr_qubits
        for i in range(groupAddr_qubits):
            qml.CNOT(wires=[i+1, i+1+groupAddr_qubits])
        # fullAnsz on qubits [groupAddr_qubits + 1, mnqubits]
        for i in range(mnlayers):
            for j in range(groupData_qubits):
                qml.RY(out_params[i, j], wires=j + groupAddr_qubits + 1)
            nCNOT2(groupAddr_qubits + 1, mnqubits)
        controlled_Unitary()
        return qml.probs(wires=range(0,1+mnqubits))

    out = circuit(out_params) # 1 + mnqubits
    prob_1 = np.sum(out[2**mnqubits:2**(mnqubits+1)])
    result = np.sqrt(out[2**mnqubits:2**(mnqubits+1)]) / np.sqrt(prob_1)

    # qml.draw_mpl(circuit, expansion_strategy="device")(out_params)
    # plt.show()
    return result

### Run the final QRAM circuit and obtain the final quantum state as psi_Final_QRAM
psi_Final_QRAM = QRAM_Final(params_U1, sqrt_S, time)

# ---------------------------------------【Plot the quantum state】---------------------------------
def plotResult():
    def plotOriginaldata():
        data = psi_original_data.reshape(realDataNum, realDataDim)

        plt.subplot(2, 3, 1)
        plt.imshow(data, cmap='gray')
        plt.title('(1) Original_data')

    def plotEmbeddeddata():
        data = psi_data.reshape(groupNum, groupDim)

        plt.subplot(2, 3, 2)
        plt.imshow(data, cmap='gray')
        plt.title('(2) Grouped_data')

    def plotOrthogonalData():
        data_Orth = psi_dataTilde.reshape(groupNum, groupDim)

        plt.subplot(2, 3, 3)
        plt.imshow(data_Orth, cmap='gray')
        plt.title('(3) data_Orth_byS')

    def plotQRAMOrthogonal():
        ##  Plot the quantum state after "Stage 2: QRAM circuit for orthogonalized data"
        Orth_QRAM = psi_Orth_QRAM.reshape(groupNum, groupDim)

        plt.subplot(2, 3, 6)
        plt.imshow(Orth_QRAM, cmap='gray')
        plt.title('(4) Orth_QRAM')

    def plotmidQRAM():
        ##  Plot the grouped quantum state after "Stage 3: QRAM circuit for the reverse transformation"
        Final_QRAM = psi_Final_QRAM.reshape(groupNum, groupDim)

        plt.subplot(2, 3, 5)
        plt.imshow(Final_QRAM, cmap='gray')
        plt.title('(5) Grouped_QRAM')

    def plotfinalQRAM():
        ##  Plot the final quantum state after "Stage 3: QRAM circuit for the reverse transformation"
        Final_QRAM = psi_Final_QRAM.reshape(realDataNum, realDataDim)

        plt.subplot(2, 3, 4)
        plt.imshow(Final_QRAM, cmap='gray')
        plt.title('(6) Final_QRAM')

    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plotOriginaldata()
    plotEmbeddeddata()
    plotOrthogonalData()
    plotQRAMOrthogonal()
    plotmidQRAM()
    plotfinalQRAM()
    plt.show()

plotResult()

fidelity_orth = np.dot(psi_dataTilde, psi_Orth_QRAM)
fidelity_final = np.dot(psi_data, psi_Final_QRAM)
print('fidelity_orth = ', fidelity_orth)
print('fidelity_final = ', fidelity_final)
print('End.\n')
