import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        scalProd = nn.as_scalar(self.run(x))
        if scalProd >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        mistakeCheck = True
        while mistakeCheck:
            mistakeCheck = False
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    direction = nn.Constant(x.data * nn.as_scalar(y))
                    self.w.update(direction, 1)
                    mistakeCheck = True


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.wOne = nn.Parameter(1, 100)
        self.bOne = nn.Parameter(1, 100)
        self.wTwo = nn.Parameter(100, 50)
        self.bTwo = nn.Parameter(1, 50)
        self.wThree = nn.Parameter(50, 1)
        self.bThree = nn.Parameter(1, 1)
        self.learningRate = -.01
        self.variables = [self.wOne, self.bOne, self.wTwo, self.bTwo, self.wThree, self.bThree]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        firstLinear = nn.Linear(x, self.wOne)
        firstLayer = nn.ReLU(nn.AddBias(firstLinear, self.bOne))
        secondLinear = nn.Linear(firstLayer, self.wTwo)
        secondLayer = nn.ReLU(nn.AddBias(secondLinear, self.bTwo))
        thirdLinear = nn.Linear(secondLayer, self.wThree)
        thirdLayer = nn.AddBias(thirdLinear, self.bThree)
        return thirdLayer

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        yHat = self.run(x)
        return nn.SquareLoss(yHat, y)
        
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        loss = float('inf')
        while loss >= .02:
            for x, y in dataset.iterate_once(100):
                nextLoss = self.get_loss(x, y)
                loss = nn.as_scalar(nextLoss)
                gradients = nn.gradients(nextLoss, self.variables)
                for i in range(0, 6):
                    self.variables[i].update(gradients[i], self.learningRate)



class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        self.wOne = nn.Parameter(784, 300)
        self.bOne = nn.Parameter(1, 300)
        self.wTwo = nn.Parameter(300, 150)
        self.bTwo = nn.Parameter(1, 150)
        self.wThree = nn.Parameter(150, 75)
        self.bThree = nn.Parameter(1, 75)
        self.wThree = nn.Parameter(150, 75)
        self.bThree = nn.Parameter(1, 75)
        self.wFour = nn.Parameter(75, 10)
        self.bFour = nn.Parameter(1, 10)
        self.learningRate = -.45
        self.variables = [self.wOne, self.bOne, self.wTwo, self.bTwo, self.wThree, self.bThree, self.wFour, self.bFour]
        "*** YOUR CODE HERE ***"

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        firstLinear = nn.Linear(x, self.wOne)
        firstLayer = nn.ReLU(nn.AddBias(firstLinear, self.bOne))
        secondLinear = nn.Linear(firstLayer, self.wTwo)
        secondLayer = nn.ReLU(nn.AddBias(secondLinear, self.bTwo))
        thirdLinear = nn.Linear(secondLayer, self.wThree)
        thirdLayer = nn.ReLU(nn.AddBias(thirdLinear, self.bThree))
        fourthLinear = nn.Linear(thirdLayer, self.wFour)
        fourthLayer = nn.AddBias(fourthLinear, self.bFour)
        return fourthLayer

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        yHat = self.run(x)
        return nn.SoftmaxLoss(yHat, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        loss = float('inf')
        accuracy = 0
        while accuracy < .98:
            for x, y in dataset.iterate_once(50):
                nextLoss = self.get_loss(x, y)
                loss = nn.as_scalar(nextLoss)
                gradients = nn.gradients(nextLoss, self.variables)
                for i in range(0, 8):
                    self.variables[i].update(gradients[i], self.learningRate)
            accuracy = dataset.get_validation_accuracy()
            self.learningRate /= 2

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.intialW = nn.Parameter(47, 100)
        self.initialB = nn.Parameter(1, 100)
        self.xw = nn.Parameter(47, 100)
        self.hw = nn.Parameter(100, 100)
        self.b = nn.Parameter(1, 100)
        self.wLang = nn.Parameter(100, 5)
        self.bLang = nn.Parameter(1, 5)
        self.learningRate = -.25
        self.variables = [self.intialW, self.initialB, self.xw, self.hw, self.b, self.wLang, self.bLang]

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        nodeLayer = nn.AddBias(nn.Linear(xs[0], self.intialW), self.initialB)
        for node in xs[1:]:
            addNode = nn.Add(nn.Linear(node, self.xw), nn.Linear(nn.ReLU(nodeLayer), self.hw))
            nodeBias = nn.AddBias(addNode, self.b)
            nodeLayer = nn.ReLU(nodeBias)
        finalLinear = nn.Linear(nodeLayer, self.wLang)
        finalLayer = nn.AddBias(finalLinear, self.bLang)
        return finalLayer

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        yHat = self.run(xs)
        return nn.SoftmaxLoss(yHat, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        loss = float('inf')
        accuracy = 0
        while accuracy < .81:
            for x, y in dataset.iterate_once(10):
                nextLoss = self.get_loss(x, y)
                loss = nn.as_scalar(nextLoss)
                gradients = nn.gradients(nextLoss, self.variables)
                for i in range(0, len(self.variables)):
                    self.variables[i].update(gradients[i], self.learningRate)
            accuracy = dataset.get_validation_accuracy()
            self.learningRate /= 20 