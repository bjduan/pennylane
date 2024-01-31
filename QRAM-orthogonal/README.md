## Install Pennylane
PennyLane is a cross-platform Python library for differentiable programming of quantum computers.
For detailed installation instructions, please refer to the official tutorial available at 
https://pennylane.ai/install/.

## Data Availability
Our examples utilize two datasets: the Handwritten Digits dataset and the Iris dataset. 
These datasets are sourced from scikit-learn and can be accessed at scikit-learn Dataset Examples:
https://scikit-learn.org/stable/auto_examples/index.html#dataset-examples.

## Example Tutorials
We provide two example tutorials that can be executed directly: 
'qram-digits-m2n6.py' and 'qram-iris-m4n2.py'.

### Example-1. qram-digits-m2n6.py

Each datapoint in the handwritten digits dataset is represented as an $8\times 8$ image of a digit, 
which is then transformed into an $N$=64 dimensional vector. 
Loading a single data point requires $n=\log_2 N=6$ qubits. 
In this tutorial, we choose $M=4$ datapoints, implying that $m=\log_2 M=2$.

Users can customize parameters in the designated user-input sections. 
For instance, if loading 32 data points, the number of qubits (nqubits) should be set to 5.

### Example-2. qram-iris-m4n2.py

The dimensionality of each datapoint of the Iris dataset is $N=4$, resulting in $n=\log_2 N=2$ qubits. 
In this tutorial, we choose $M=16$ datapoints, implying that $m=\log_2 M=4$.
The variables 'realDataNum' and 'realDataDim' represent the original data number and data dimension, respectively. 
Similarly, 'groupNum' and 'groupDim' represent the number and dimension of the grouped-data.

It is important to note that we can adjust groupNum and groupDim with different settings, 
ensuring only that 'realDataNum * realDataDim' = 'groupNum * groupDim'.

In both examples, we train parameters for the QRAM circuit and store the trained parameters in a '.pkl' file. 
Feel free to explore and experiment by modifying the example scripts and adjusting the parameters as needed.
