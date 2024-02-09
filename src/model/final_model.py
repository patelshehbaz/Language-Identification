import torch.nn as nn
from src.model.base_model import ImageClassification

class CNNNetwork(ImageClassification):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = self.conv_block(in_channels, 16, pool=True) 
        self.conv2 = self.conv_block(16, 32, pool=True) 
        self.conv3 = self.conv_block(32, 64, pool=True) 
        self.conv4 = self.conv_block(64, 128, pool=True) 
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(128 * 5 * 4, num_classes)
        self.softmax = nn.Softmax(dim=1)
  
    @staticmethod
    def conv_block(in_channels, out_channels, pool=False):
        """
        > The function takes in two arguments, `in_channels` and `out_channels`, and returns a sequential
        container of layers. 
        
        The sequential container is a list of layers that are executed in sequence. The first layer is a
        convolutional layer with a kernel size of 3, stride of 1, and padding of 2. The second layer is a
        batch normalization layer, and the third layer is a ReLU activation function. If the `pool` argument
        is set to `True`, then a max pooling layer is added to the sequential container. 
        
        The `*` in front of `layers` in the return statement is a Python operator that unpacks the list of
        layers. 

        Args:
          in_channels: number of input channels
          out_channels: number of channels in the output image
          pool: if True, then we add a MaxPool2d layer to the end of the block. Defaults to False
        
        Returns:
          A sequential object that contains the layers of the convolutional block.
        """
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2), 
                  nn.BatchNorm2d(out_channels), 
                  nn.ReLU(inplace=True)]
        if pool: layers.append(nn.MaxPool2d(kernel_size=2))
        return nn.Sequential(*layers) 

    def forward(self, input_data):
        """
        The forward function takes in an input image, passes it through a series of convolutional and max
        pooling layers, and then through a fully connected layer
        
        Args:
          input_data: The input data to the network.
        
        Returns:
          The predictions of the model.
        """
        out = self.conv1(input_data)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.flatten(out)
        out = self.dropout(out)
        logits = self.linear(out)
        predictions = self.softmax(logits)
        return predictions