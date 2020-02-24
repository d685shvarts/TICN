import torch
import torch.nn as nn
import torchvision

class LSTM(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        self.relu = nn.ReLU(inplace=True)
        
        # Resnet50 of encoder
        self.resnet = torchvision.models.resnet50(pretrained=True)

        # Freeze Resnet50
        for layer in self.resnet.children():
            for param in layer.parameters():
                param.requires_grad = False
        
        # Replace last FC with our our own linear feature
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, cfg['embedding_size'])
        self.resnetBN = nn.BatchNorm1d(cfg['embedding_size'])
        
        
        # Decoder LSTM
        self.lstm = nn.LSTM(input_size=cfg['embedding_size'], hidden_size=cfg['embedding_size'], num_layers=cfg['num_layers'], batch_first=True, dropout=cfg['dropout'], bidirectional=cfg['bidirectional'])


    def forward(self, X=None, imgs=None, hidden_state=None, cell_state=None):
        '''
        X -- (batch_size, sequence_length, embedding_size)
        imgs -- not None with dimension (batch_size, img_height, img_width)
        hidden_state -- (hidden_size)
        cell_state -- (cell_size)
        
        There are two modes to use forward:
        1. For learning: The whole sequence of teacher is passed in.
            For any batch, we just need to pass in X as a whole teacher sequence and the imgs,
            please ignore hidden_state and cell_state
                
        2. For generation: We need to call forward multiple times, each time passing in Xi and producing Xi+1,
        which is again used as input to subsequent forward call.
            For the first forward call of an image (or a batch), only imgs should be passed in
            For subsequent calls, only pass in X, hidden_state and cell_state generated from previous call
        '''

        # For generation, imgs can be None if we want to pass in X1, X2, X3 in separate calls
        # In such cases, it only needs to be not None for the first call of a batch
        if imgs is not None:

            # Make sure img is dimension (batch_size, img_height, img_width)
            if (len(imgs.shape) != 4):
                raise ValueError("Expecte imgs to have 3 dimensions, got {}".format(imgs.shape))

            # Create feature vector: (batch_size, img_height, img_width) --> (batch_size, embedding_size)
            X0 = self.resnetBN(self.relu(self.resnet(imgs)))

            # Make this output into 3D (batch_size, embedding_size) --> (batch_size, 1, embedding_size)
            X0 = X0.view((X0.shape[0], 1, X0.shape[-1]))

        # For generation, X can be None in the first forward call if we want to pass in X1, X2, X3
        # in separate subsequent calls
        if imgs is not None and X is not None:
            # Stack encoder's feature vector to be the first of each sequence
            X = torch.cat([X0, X], dim=1)
        elif X is None:
            X = X0
        elif imgs is None:
            X = X
        
                            
        # For generation, hidden_state and cell_state needs to be not None if we want to pass in 
        # X1, X2, X3 in separate subsequent calls
        if hidden_state is not None and cell_state is not None:
            # LSTM the sequence
            Y, (hidden_state, cell_state) = self.lstm(X, (hidden_state, cell_state))
        else:
            # LSTM the sequence
            Y, (hidden_state, cell_state) = self.lstm(X)

        return Y, (hidden_state, cell_state)
                        