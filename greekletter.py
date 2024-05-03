# CS 5330 - Computer Vision
# Project 5
# Name: Pranav Raj Sowrirajan Balaji
# Date: 26 March 2024
# Purpose : To train a neural network and transfer learning onto Greek letters
# Tasks: 
# 3. Transfer learning onto Greek letters
# A. Plot of training errors B. Test the network on new images


# Import the required libraries
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

# Define the GreekTransform class
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = transforms.functional.rgb_to_grayscale(x)
        x = transforms.functional.affine(x, 0, (0,0), 36/128, 0)
        x = transforms.functional.center_crop(x, (28, 28))
        return transforms.functional.invert(x)

# Define the Net class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Define the main function
def main(argv):

    # Load the trained network
    network = Net()
    network.load_state_dict(torch.load('/Users/pranavraj/Desktop/visualstudio-opencv/results/model.pth'))

    # Freeze the network weights
    for param in network.parameters():
        param.requires_grad = False

    # Replace the last layer with a new Linear layer with three nodes
    network.fc2 = nn.Linear(50, 3)

    # DataLoader for the Greek data set
    training_set_path = '/Users/pranavraj/Desktop/visualstudio-opencv/files/greek_train/'
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(training_set_path,
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                    GreekTransform(),
                                                                    transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=27,
        shuffle=True)

    # Train the network on the Greek letters
    optimizer = torch.optim.SGD(network.fc2.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Initialize a list to store the loss values for each epoch
    training_errors = []
    n = len(greek_train)  # get the number of batches in the dataset

    for epoch in range(100):  # loop over the dataset multiple times

        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(greek_train, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # print statistics
            running_loss += loss.item()
            if i % n == n-1:    # print every n mini-batches
                average_loss = running_loss / n
                accuracy = 100 * correct / total
                print('[%d, %5d] loss: %.3f, accuracy: %.2f%%' % (epoch + 1, i + 1, average_loss, accuracy))
                training_errors.append(average_loss)
                running_loss = 0.0
                correct = 0
                total = 0

    print('Finished Training')

    # Plot the training error
    
    plt.plot(training_errors)
    plt.title('Training Error')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    print('Final training error: %.3f' % training_errors[-1])

    image_paths = ['/Users/pranavraj/Desktop/visualstudio-opencv/Mygreekletters/alpha/IMG_1668.jpg',
                   '/Users/pranavraj/Desktop/visualstudio-opencv/Mygreekletters/beta/IMG_1679.jpg',
                   '/Users/pranavraj/Desktop/visualstudio-opencv/Mygreekletters/gamma/IMG_1675.jpg'
                   ]
    

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize(128),  # Resize the short side of the image to 128 keeping aspect ratio
        transforms.CenterCrop(128),  # Crop a square in the center of the image
        transforms.ToTensor(),  # Convert the image to PyTorch Tensor data type
        GreekTransform(),  # Apply the GreekTransform
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize the image
    ])

    # Load and classify the images
    for image_path in image_paths:
        image = Image.open(image_path)
        image = transforms.ToTensor()(image)
        image = GreekTransform()(image)
        image = transforms.Normalize((0.1307,), (0.3081,))(image)
        image = image.unsqueeze(0)  # add a batch dimension

        # Forward pass through the network
        output = network(image)
        _, predicted = torch.max(output.data, 1)

        # Print the predicted class
        print(f'Predicted class for {image_path}: {predicted.item()}')

if __name__ == "__main__":
    main(sys.argv)
