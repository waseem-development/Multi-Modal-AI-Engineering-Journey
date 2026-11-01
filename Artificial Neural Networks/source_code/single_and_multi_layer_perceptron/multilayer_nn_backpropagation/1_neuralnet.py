import numpy as np

class NeuralNetwork:

    def __init__(self, layers, alpha=0.1):
        #initialize the weight matrix and 
        #store the network architecture and learning rate
        self.W = []
        self.layers = layers
        self.alpha = alpha

        for i in np.arange(0, len(layers)-2):
            w = np.random.randn(layers[i] + 1, layers[i+1] + 1)
            self.W.append( w / np.sqrt(layers[i]))

        # The last two layers are handled separately 
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))
    
    def __repr__(self):
        return "NeuralNetwork: {}".format(
                "-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        # compute and return the sigmoid activation value for a
        return 1.0 / (1 + np.exp(-x))
    def sigmoid_deriv(self, x):
        # compute the derivative of the sigmoid function 
        return x * (1-x)

    def fit(self, X, y, epochs=1000, displayUpdate=100):
        # insert a column of 1’s as the last entry in the feature
        X = np.c_[X, np.ones((X.shape[0]))]
        
        for epoch in np.arange(0, epochs):
            # loop over each individual data point and train
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(
                    epoch + 1, loss))

    def fit_partial(self, x, y):
        # construct our list of output activations for each layer
        A = [np.atleast_2d(x)]

        #***** FEED FORWARD *****
        # loop over the layers in the network
        for layer in np.arange(0, len(self.W)):
            # feedforward the activation at the current layer by
            # taking the dot product between the activation and
            # the weight matrix -- this is called the "net input"
            # to the current layer
            net = A[layer].dot(self.W[layer])
            # computing the "net output" is simply applying our
            # nonlinear activation function to the net input
            out = self.sigmoid(net)

            # once we have the net output, add it to our list of
            # activations
            A.append(out)
            #print(layer)
        
        #****** BACKPROPAGATION ********
        error = A[-1] - y
        # * From here, we need to apply the chain rule and build our
        # list deltas 'D'
        # * The first entry in the deltas is simply the error of the
        # output layer times the derivative of the activation function 
        # for the output value
        D = [error * self.sigmoid_deriv(A[-1])]
        # The chain rule:: simply loop over the layers in reverse
        # order (ignoring the last two, as we already take them into
        # account 
        for layer in np.arange(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)
        
        #Since we looped over our layers in reverse order we need
        #reverse the detlas
        D=D[::-1]
        #print(D[2])
        #**** WEIGHT UPDATE ***#
        for layer in np.arange(0, len(self.W)):
            ### Some comments here
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])
    
    def predict(self, X, addBias=True):
        # Initialize the output prediction as the input features, this
        # value will be (forward) propagated through the network to
        # obtain the final prediction
        p = np.atleast_2d(X)

        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.W)):
            # as simple as 
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p
      
    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss


#X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#y = np.array([[0], [1], [1], [0]])
#
#nn = NeuralNetwork([2, 2, 1], alpha=0.5)
#print(nn)
#nn.fit(X, y, epochs=2000)
#
#for (x, target) in zip(X, y):
##    print(nn.predict(x)[0])
#    pred = nn.predict(x)[0][0]
#    step=1 if pred > 0.5 else 0
#    print("[INFO] data={}, ground_truth={}, pred={:.4f},step={}".format(x,
#        target[0], pred, step))
#
