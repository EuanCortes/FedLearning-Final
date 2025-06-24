from typing import List, Tuple
import torch

class SmallCNN(torch.nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()

        self.relu = torch.nn.ReLU()               # activation function

        # convcolutional layers
        self.ConvLayers = torch.nn.ModuleList([
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),   # N x 3 x 32 x 32 -> N x 32 x 32 x 32
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),  # N x 32 x 16 x 16 -> N x 64 x 16 x 16
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1), # N x 64 x 8 x 8 -> N x 128 x 8 x 8
        ])

        self.fc1 = torch.nn.Linear(128 * 4 * 4, 256)      # first fully connected layer after convolutional layers
        self.fc2 = torch.nn.Linear(256, 10)               # final fully connected layer for output

        self.pool = torch.nn.MaxPool2d(2, 2)      # max pooling layer for regularization
        self.dropout = torch.nn.Dropout(0.2)      # dropout layer for regularization

    def forward(self, x):
        for conv in self.ConvLayers:
            # apply convolutional layer, then batch normalization, then ReLU, then max pooling
            x = conv(x)
            x = self.pool(self.relu(x))       # final shape: N x 128 x 4 x 4

        x = x.view(x.size(0), -1)       # reshape to N x 128*4*4
        x = self.relu(self.fc1(x))      # fully connected layer and ReLU
        x = self.dropout(x)             # apply dropout for regularization
        x = self.fc2(x)                 # final fully connected layer for output
        return x
    


################################## Training using regular descent steps ##################################
def train(
        net: torch.nn.Module, 
        device: torch.device,
        trainloader: torch.utils.data.DataLoader, 
        criterion: torch.nn.Module,
        num_epochs: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        optim: str = "sgd",
        ) -> None:
    """ 
    function that trains a model on the training dataset.
    """
    net.to(device)

    if optim.lower() == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim.lower() == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        print(f"Unknown optimizer: {optim}. Using SGD by default.")
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    net.train()
    for epoch in range(num_epochs):
        for batch in trainloader:
            images, labels = batch["img"], batch["label"]
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()



#################################### Scaffold Optimizer ##################################
class ScaffoldOptimizer(torch.optim.SGD):
    def __init__(self, params, lr, momentum=0., weight_decay=0.):
        super().__init__(params, lr, momentum, weight_decay)

    def step_custom(self, global_cv, client_cv):
        """
        Perform a single optimization step.
        :param global_cv: Global control variable
        :param client_cv: Client control variable
        """
        # compute regular SGD step
        #   w <- w - lr * grad
        super().step() 

        # now add the correction term
        #   w <- w - lr * (g_cv - c_cv)
        device = self.param_groups[0]["params"][0].device
        for group in self.param_groups:
            for param, g_cv, c_cv in zip(group["params"], global_cv, client_cv):
                # here we add the correction term to each parameter tensor.
                # the alpha value scales the correction term
                    g_cv, c_cv = g_cv.to(device), c_cv.to(device)
                    param.data.add_(g_cv - c_cv, alpha=-group["lr"]) 
                #if param.grad is not None:
                    #g_cv, c_cv = g_cv.to(device), c_cv.to(device)
                    #param.grad.add_(g_cv - c_cv)  #, alpha=-group["lr"]) 
        #super().step()


##################################### Training using Scaffold updates ##################################
def train_scaffold(net: torch.nn.Module, 
                   device: torch.device, 
                   trainloader: torch.utils.data.DataLoader,
                   criterion: torch.nn.Module,
                   num_epochs: int, 
                   lr: float, 
                   momentum: float, 
                   weight_decay: float, 
                   global_cv: List[torch.Tensor], 
                   client_cv: List[torch.Tensor],
                   ) -> None:
    """
    Function that trains a model using the Scaffold optimization algorithm.
    Parameters:
        net:            The neural network model to train.
        device:         The device to run the training on (CPU or GPU).
        trainloader:    DataLoader for the training data.
        criterion:      Loss function to use for training.
        num_epochs:     Number of epochs to train the model.
        lr:             Learning rate for the optimizer.
        momentum:       Momentum factor for the optimizer.
        weight_decay:   Weight decay (L2 penalty) for the optimizer.
        global_cv:      Global control variables for Scaffold.
        client_cv:      Client control variables for Scaffold.
    """
    net.to(device)
    optimizer = ScaffoldOptimizer(
        net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    net.train()
    
    for _ in range(num_epochs):
        for batch in trainloader:
            Xtrain, Ytrain = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            output = net(Xtrain)
            loss = criterion(output, Ytrain)

            # for debugging purposes, exit if loss is NaN
            #if torch.isnan(loss):
            #    raise ValueError("Loss is NaN, check your model and data.")

            loss.backward()
            
            # Perform a single optimization step with the control variables
            optimizer.step_custom(global_cv, client_cv)




################################### Testing the model ##################################
def test(net: torch.nn.Module, 
         device: torch.device, 
         testloader: torch.utils.data.DataLoader,
         criterion: torch.nn.Module,
         ) -> Tuple[float, float]:
    """
    Function that tests a model on the test dataset.
    Parameters:
        net:        The neural network model to test.
        device:     The device to run the testing on (CPU or GPU).
        testloader: DataLoader for the test data.
        criterion:  Loss function to use for testing.
    Returns:
        Tuple containing the average loss and accuracy on the test set.
    """
    
    net.eval()
    total_loss = 0.0    # Accumulator for total loss
    correct = 0         # tracker for correct predictions
    total = 0           # tracker for total predictions
    
    with torch.no_grad():
        for batch in testloader:
            Xtest, Ytest = batch["img"].to(device), batch["label"].to(device)
            output = net(Xtest)
            loss = criterion(output, Ytest)

            if torch.isnan(loss):
                raise ValueError("Loss is NaN, check your model and data.")

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += Ytest.size(0)
            correct += predicted.eq(Ytest).sum().item()
    
    avg_loss = total_loss / len(testloader) # compute the average loss
    accuracy = correct / total              # compute the accuracy
    return avg_loss, accuracy