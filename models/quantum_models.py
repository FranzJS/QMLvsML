import pennylane as qml
import torch.nn as nn
from pennylane.templates import StronglyEntanglingLayers



class QuantumModel(nn.Module):
    def __init__(self, input_dim, n_layers=1, n_trainable_block_layers=3):
        super().__init__()
        self.n_qubits = input_dim
        self.n_layers = n_layers
        self.n_trainable_block_layers = n_trainable_block_layers
        self.device = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.device)
        def circuit(inputs, weights):

            for theta in weights[:-1]:
                StronglyEntanglingLayers(theta, wires=range(self.n_qubits)) # W
                qml.AngleEmbedding(inputs, wires=range(self.n_qubits))

            StronglyEntanglingLayers(weights[-1], wires=range(self.n_qubits)) # (L+1)'th W-block

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        # n_layers+1 is due to our notation W^{(L)}...W^{(0)}
        weight_shapes = {"weights" : (self.n_layers+1, self.n_trainable_block_layers, self.n_qubits, 3)} 
        self.qcircuit = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        return self.qcircuit(x)



class QuantumRegressionModel(QuantumModel):
    def __init__(self, input_dim, n_layers=1, n_trainable_block_layers=3):
        super().__init__(input_dim, n_layers, n_trainable_block_layers)

        self.Linear = nn.Linear(self.n_qubits, 1, bias=False).double()

    def forward(self, x):
        x = self.qcircuit(x)
        x = self.Linear(x) # note that we do not freeze this linear layer as a default!
        return x



        
class QuantumClassificationModel(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=1, n_trainable_block_layers=3):
        super().__init__()
        self.n_qubits = input_dim
        self.n_classes = output_dim
        self.n_layers = n_layers
        self.n_trainable_block_layers = n_trainable_block_layers
        self.device = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.device)
        def circuit(inputs, weights):

            for theta in weights[:-1]:
                StronglyEntanglingLayers(theta, wires=range(self.n_qubits)) # W
                qml.AngleEmbedding(inputs, wires=range(self.n_qubits))

            StronglyEntanglingLayers(weights[-1], wires=range(self.n_qubits)) # (L+1)'th W-block

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_classes)]


        weight_shapes = {"weights" : (self.n_layers, self.n_trainable_block_layers, self.n_qubits, 3)}
        self.qcircuit = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        return self.qcircuit(x)