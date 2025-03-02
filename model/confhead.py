from torch import nn

class ConfidenceHeadFlattenInput(nn.Module):
    def __init__(self, hidden_size):
        super(ConfidenceHeadFlattenInput, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, hidden_state):
        x = self.linear(hidden_state)
        conf = self.sigmoid(x)
        return conf.squeeze(-1)
    

class ConfidenceHeadFlattenInputAug(nn.Module):
    def __init__(self, hidden_size):
        super(ConfidenceHeadFlattenInputAug, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 2)
        self.linear2 = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, hidden_state):
        x = self.linear1(hidden_state)
        x = nn.ReLU()(x)
        x = self.linear2(x)
        conf = self.sigmoid(x)
        return conf.squeeze(-1)
        

class ConfidenceHeadStackedInput(nn.Module):
    def __init__(self, num_heads, head_dim, 
                 conv_channels=32, 
                 kernel_size=1, 
                 conv_activation='relu',
                 dropout=0.0):
        super(ConfidenceHeadStackedInput, self).__init__()
        self.conv = nn.Conv1d(in_channels=num_heads, out_channels=conv_channels, kernel_size=kernel_size)
        self.activation = get_activation(conv_activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(conv_channels, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, per_head_hidden):
        # Apply conv1d: output shape (batch, conv_channels, new_length)
        conv_out = self.conv(per_head_hidden)
        conv_out = self.activation(conv_out)
        conv_out = self.dropout(conv_out)
        # Pool over the length dimension so that output becomes (batch, conv_channels, 1)
        pooled = self.pool(conv_out)
        pooled = pooled.squeeze(-1)  # Shape: (batch, conv_channels)
        out = self.fc(pooled)         # Shape: (batch, 1)
        conf = self.sigmoid(out)      # Shape: (batch, 1)
        return conf.squeeze(-1)       # Shape: (batch,)


def get_activation(activation_name: str):
    if activation_name == 'relu':
        return nn.ReLU()
    elif activation_name == 'gelu':
        return nn.GELU()
    elif activation_name == 'tanh':
        return nn.Tanh()
    elif activation_name == 'sigmoid':
        return nn.Sigmoid()
    elif activation_name == 'selu':
        return nn.SELU()
    else:
        raise ValueError(f"Unsupported conv activation: {activation_name}")