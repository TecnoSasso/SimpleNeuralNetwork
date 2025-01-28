import random
import math
from PIL import Image

lines = []
examples = []
labels = []
testData = []
testLabels = []

print("Loading training data...")

# Read the examples of the training data
with open("mnist_train.csv", "r") as f:
    for i, l in enumerate(f):
        if i >= 60000:
            break

        lines.append(l.split(","))

for l in lines:
    labels.append(int(l[0]))
    example = []
    for i in l[1:]:
        example.append(int(i)/255)

    examples.append(example)

if len(examples) != len(labels):
    raise ValueError

lines = []
example = []

# Read the examples of the test data
with open("mnist_test.csv", "r") as f:
    for i, l in enumerate(f):
        if i >= 1000:
            break

        lines.append(l.split(","))

for l in lines:
    testLabels.append(int(l[0]))
    example = []
    for i in l[1:]:
        example.append(int(i)/255)

    testData.append(example)

lines = []
example = []


def Sigmoid(x):
    return max(0, x)
    """
    try:
        return 1/(1 + math.e**(-x))  # Sigmoid
    except OverflowError:
        if x > 0:
            return 1
        else:
            return 0
    """

def SigmoidDer(x):
    return 1 if x > 0 else 0
    """
    s = Sigmoid(x)
    return s*(1-s)
    """


class neuron:
    def __init__(self):
        self.inConnections = []
        self.outConnections = []
        self.bias = 0
        self.value = 0
        self.z = 0
        self.influence = 0
        self.biasInfluence = []
    def evaluate(self):
        self.value = 0
        # Weighted sum
        for c in self.inConnections:
            self.value += c.start.value * c.weight
        self.value += self.bias
        self.z = self.value
        self.value = Sigmoid(self.value)

class connection():
    def __init__(self, start, end, n_in):
        self.influence = []
        self.weight = random.gauss(0, math.sqrt(2/n_in))
        self.start = start
        self.end = end
        self.end.inConnections.append(self)
        self.start.outConnections.append(self)

def printNetwork(exampleIndex):
    o = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    o[labels[exampleIndex]] = 1
    print("----  EXPECTED OUTPUT  ----")
    print(o)

    print("----  OUTPUT NEURONS VALUES  ----")
    for n in Layers[len(Layers)-1]:
        print(n.value)

def feedForward(input):
    global Layers
    # Putting inputs values
    for j, n in enumerate(inputLayer):
        n.value = input[j]

    # Feeding forward
    for i in range(len(Layers)):
        if (i == 0):
            continue
        else:
            for n in Layers[i]:
                n.evaluate()

def cost(exampleIndex):
    global Layers
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y[labels[exampleIndex]] = 1
    cost = 0
    for i, n in enumerate(outputLayer):
        cost += (n.value - y[i])**2
    return cost

# Derivative of Cost with respect to a weight
def DerCostWeight(connection, y):
    if len(connection.end.outConnections) == 0:  # If the connection is in the last layer
        yI = outputLayer.index(connection.end)
        return connection.start.value*SigmoidDer(connection.end.z)*2*(connection.end.value - y[yI])  # d/dW C

    else:  # Not a connection in the last layer
        return connection.start.value*SigmoidDer(connection.end.z)*connection.end.influence  # d/dW A * d/dA C


# Derivative of Cost with respect to a bias
def DerCostBias(Neuron, y):
    if len(Neuron.outConnections) == 0:  # If the neuron is in the last layer
        yI = outputLayer.index(Neuron)
        return SigmoidDer(Neuron.z)*2*(Neuron.value - y[yI])  # d/dB C

    else:  # Not a connection in the last layer
        return SigmoidDer(Neuron.z)*Neuron.influence  # d/dB A * d/dA C


# Derivative of Cost with respect to the neuron activation
def DerCostAct(Neuron, y):
    finalDer = 0
    # If the Neuron is in the second to last layer
    if len(Neuron.outConnections[0].end.outConnections) == 0:
        for c in Neuron.outConnections:
            yI = outputLayer.index(c.end)
            finalDer += c.weight*SigmoidDer(c.end.z)**2*(c.end.value - y[yI])

    else:  # Not a connection in the second to last layer
        for c in Neuron.outConnections:
            finalDer += c.weight*SigmoidDer(c.end.z)*c.end.influence  # d/dA A[L+1] * d/A[L+1] C
        
    return finalDer


def backPropagation(number):
    global Layers
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y[labels[number]] = 1

    for l in Layers[::-1]:
        for n in l:
            if l == inputLayer:
                continue
            
            # First i find the influence of the neuron activation on the bias
            if l != outputLayer:
                n.influence = DerCostAct(n, y)

            # Then the influence of the connections and the bias
            for c in n.inConnections:
                influence = DerCostWeight(c, y)
                c.influence.append(influence)

            biasInfluence = DerCostBias(n, y)
            n.biasInfluence.append(biasInfluence)

def trainNetwork(batchSize = 10, step = 0.1):
    global Layers
    global Connections
    averageCost = 0
    for c in Connections:
        c.influence = []  # reset the gradient
    for n in Neurons:
        n.biasInfluence = []  # reset the bias gradient
    
    exampleIndex = 0

    # Finding the gradient -------
    for i in range(batchSize):
        exampleIndex = random.randint(0, len(examples)-1)
        example = examples[exampleIndex]

        feedForward(example)
        averageCost += cost(exampleIndex)
        backPropagation(exampleIndex)
    
    # Taking the step downhill -------
    for c in Connections:
        c.weight -= sum(c.influence) / len(c.influence) * step  # average of the influence * step

    for l in Layers:
        if l == inputLayer:
            continue
        for n in l:
            n.bias -= sum(n.biasInfluence) / len(n.biasInfluence) * step

    return averageCost / batchSize


def testNetwork(exampleIndex):
    exampleTest = testData[exampleIndex]
    feedForward(exampleTest)
    outputLayerValues = [(n.value) for n in outputLayer]

    o = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    o[testLabels[exampleIndex]] = 1
    #print("EXPECTED:", ["{:.2f}".format(i) for i in o])
    #print("ACTUAL:  ", [("{:.2f}".format(n.value)) for n in outputLayer])
    #print()
    return outputLayerValues.index(max(outputLayerValues)) == testLabels[exampleIndex]


# ----------------------------------------------------------

InNeurons = 28*28  # Numbers of neurons in the input layer
HiddenNeurons = [20, 20]  # Numbers of neurons in a hidden layer
OutNeurons = 10  # Numbers of neurons in the output layer

# Create the neural network

Layers = []
Connections = []
Neurons = []

Layers.append([])
for i in range(InNeurons):
    newNeuron = neuron()
    newNeuron.bias = 0  # i don't want a constant bias in the input (yes this took a while to spot)
    Layers[0].append(newNeuron)

for i in range(len(HiddenNeurons)):
    Layers.append([])
    for j in range(HiddenNeurons[i]):
        newNeuron = neuron()
        Layers[i+1].append(newNeuron)
        Neurons.append(newNeuron)

Layers.append([])
for i in range(OutNeurons):
    newNeuron = neuron()
    Layers[len(Layers)-1].append(newNeuron)
    Neurons.append(newNeuron)

inputLayer = Layers[0]
outputLayer = Layers[len(Layers)-1]

# Create connections
for i in range(len(Layers)):
    if i+1 >= len(Layers):  # If a next layer doesn't exist
        break
    else:
        for n in Layers[i]:
            for n_out in Layers[i+1]:
                newConnection = connection(n, n_out, len(Layers[i]))
                Connections.append(newConnection)


def printText(text, coloredText, value, end="\n"):
    if value > 1:
        value = 1
    if value < 0:
        value = 0

    r = int(255 * value)
    g = int(255 * (1 - value))
    b = 0

    colorCode = f"\033[38;2;{r};{g};{b}m"
    resetCode = "\033[0m"

    print(f"{text}{colorCode}{coloredText}{resetCode}", end=end)


print("SASSO's MULTI LAYER PRECEPTRON")
print(f"Input Neurons:  {len(Layers[0])}")
print(f"Hidden Layers:  {len(Layers)-2}  Made By: {len(Layers[1])} Neurons Each")
print(f"Output Neurons: {len(Layers[len(Layers)-1])}")
totalWeights = len(Connections)
print(f"Total Connections: {totalWeights}")
totalBiases = sum(HiddenNeurons) + OutNeurons
print(f"Total Biases: {totalBiases}")
print(f"Total Dimentions: {totalWeights + totalBiases}")

print()

while True:
    mode = input("\nTrain [t], Accuracy [a], Image [i], Save [s], Load [l]: ")
    if mode == "t":
        batchSize = 0
        while batchSize <= 0:
            try:
                batchSize = int(input("Batch size: "))
                if batchSize < 0:
                    print("Batch size must be a positive number")

            except ValueError as e:
                print(e)

        gradientStep = 0
        while gradientStep <= 0:
            try:
                gradientStep = float(input("Learning rate: "))
                if gradientStep < 0:
                    print("The learning rate must be a positive number")

            except ValueError as e:
                print(e)

        while True:
            try:
                averageCost = trainNetwork(batchSize, gradientStep)
                printText("Average cost of the network: ", f"{averageCost:.4f}", averageCost)
            except KeyboardInterrupt:
                break

    elif mode == "a":
        value = 0
        testRange = 1000
        for i in range(testRange):
            value += testNetwork(i)

        print(f"TEST ACCURACY: {(value / testRange * 100):.2f}%")

    elif mode == "i":
        imageName = input("Image Name: ")
        if imageName == "":
            imageName = "canvas.jpg"

        try:
            with Image.open(f"./{imageName}", "r") as image:
                grayscale = image.convert("L")
                pixels = grayscale.load()
                inputs = []
                for y in range(28):
                    for x in range(28):
                        inputs.append(pixels[x, y]/255)
        except FileNotFoundError:
            print("File not found")


        feedForward(inputs)
        for i, n in enumerate(outputLayer):
            printText(f"{i}  ", f"{(n.value*100):.2f}%", 1 - n.value)


    elif mode == "s":
        try:
            with open("./networkData/weights.txt", "w") as weightsFile:
                weightsFile.write("\n".join([(f"{c.weight}") for c in Connections]))

            with open("./networkData/biases.txt", "w") as biasesFile:
                biasesFile.write("\n".join([(f"{n.bias}") for n in Neurons]))

        except FileNotFoundError:
            print("directory ./networkData not found, you have to create it manually")


    elif mode == "l":
        try:
            with open("./networkData/weights.txt", "r") as weightsFile:
                weightsString = weightsFile.read().split("\n")
                for i, w in enumerate(weightsString):
                    Connections[i].weight = float(w)

            with open("./networkData/biases.txt", "r") as biasesFile:
                biasesString = biasesFile.read().split("\n")
                for i, b in enumerate(biasesString):
                    Neurons[i].bias = float(b)
        
        except FileNotFoundError:
            print("Save file not found, you have to save a network first")