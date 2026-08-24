import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,SAGEConv
from torch.nn.utils.parametrizations import weight_norm
from torch_geometric.nn.norm import LayerNorm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MLP(nn.Module):
    def __init__(self, hidden_channels,input_size, output_size):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(input_size, 128)
        self.lin2 = Linear(128, hidden_channels)
        self.lin3 = Linear(hidden_channels, output_size)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin2(x)
        x = x.relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin3(x)
        return x


class GCN_test(nn.Module):
    def __init__(self, num_layers,activation,dropout,hidden_channels,input_size, output_size,use_lnorm = True):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(input_size, hidden_channels)
        self.use_lnorm = use_lnorm
        self.lnorm1 = LayerNorm(hidden_channels)
        self.activation = activation
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lnorm2 = LayerNorm(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, output_size)
        

    def forward(self, input, edge_index, mode='train'):
        if self.use_lnorm:
            x = self.lnorm1(self.conv1(input, edge_index))
        else:
            x = self.conv1(input, edge_index)
        
        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.use_lnorm:
            x = self.lnorm2(self.conv2(x, edge_index))
        else:
            x = self.conv2(x, edge_index)

        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if mode == 'visualize':
            return x
        elif mode == 'train':
            x = self.conv3(x, edge_index)
            return x


class GAT_test(nn.Module):
    def __init__(self, num_layers,activation,dropout,hidden_channels, input_size, output_size, heads,use_lnorm = True):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GATConv(input_size,hidden_channels, heads = heads, concat=False)  
        self.use_lnorm = use_lnorm
        self.lnorm1 = LayerNorm(hidden_channels)
        self.activation = activation
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.lnorm2 = LayerNorm(hidden_channels)
        self.conv3 = GATConv(hidden_channels, output_size)
        

    def forward(self, input, edge_index, mode='train'):
        if self.use_lnorm:
            x = self.lnorm1(self.conv1(input, edge_index))
        else:
            x = self.conv1(input, edge_index)
        
        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        elif self.activation == 'elu':
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.use_lnorm:
            x = self.lnorm2(self.conv2(x, edge_index))
        else:
            x = self.conv2(x, edge_index)

        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        elif self.activation == 'elu':
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if mode == 'visualize':
            return x
        elif mode == 'train':
            x = self.conv3(x, edge_index)
            return x


class GCN_basic(nn.Module):
    def __init__(self, num_layers,activation,dropout,hidden_channels,input_size, output_size,use_lnorm = True):
        super().__init__()
        torch.manual_seed(1234567)
        self.use_lnorm = use_lnorm
        self.activation = activation
        self.num_layers = num_layers
        self.dropout = dropout
        if num_layers == 1:
            self.conv1 = GCNConv(input_size, output_size)
        else:
            self.conv1 = GCNConv(input_size, hidden_channels)
            self.lnorm1 = LayerNorm(hidden_channels)

        if num_layers == 6:
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.lnorm2 = LayerNorm(hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
            self.lnorm3 = LayerNorm(hidden_channels)
            self.conv4 = GCNConv(hidden_channels, hidden_channels)
            self.lnorm4 = LayerNorm(hidden_channels)
            self.conv5 = GCNConv(hidden_channels,hidden_channels)
            self.lnorm5 = LayerNorm(hidden_channels)
            self.conv6 = GCNConv(hidden_channels, output_size)
        elif num_layers == 5:
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.lnorm2 = LayerNorm(hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
            self.lnorm3 = LayerNorm(hidden_channels)
            self.conv4 = GCNConv(hidden_channels, hidden_channels)
            self.lnorm4 = LayerNorm(hidden_channels)
            self.conv5 = GCNConv(hidden_channels, output_size)
        elif num_layers == 4:
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.lnorm2 = LayerNorm(hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
            self.lnorm3 = LayerNorm(hidden_channels)
            self.conv4 = GCNConv(hidden_channels, output_size)
        elif num_layers == 3:
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.lnorm2 = LayerNorm(hidden_channels)
            self.conv3 = GCNConv(hidden_channels, output_size)
        elif num_layers == 2:
            self.conv2 = GCNConv(hidden_channels, output_size)
        

    def forward(self, input, edge_index, mode='train'):
        if self.use_lnorm and self.num_layers != 1:
            x = self.lnorm1(self.conv1(input, edge_index))
        elif not self.use_lnorm or self.num_layers == 1:       
            x = self.conv1(input, edge_index)

        if self.num_layers == 1:
            if mode == 'visualize':
                return input
            elif mode == 'train':
                return x
        
        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.num_layers == 2:
            if mode == 'visualize':
                return x
            elif mode == 'train':
                x = self.conv2(x, edge_index)
                return x

        if self.use_lnorm and self.num_layers != 2:
            x = self.lnorm2(self.conv2(x, edge_index))
        elif not self.use_lnorm or self.num_layers == 2:
            x = self.conv2(x, edge_index)
        
        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.num_layers == 3:
            if mode == 'visualize':
                return x
            elif mode == 'train':
                x = self.conv3(x, edge_index)
                return x
        
        if self.use_lnorm and self.num_layers != 3:
            x = self.lnorm3(self.conv3(x, edge_index))
        elif not self.use_lnorm or self.num_layers == 3:
            x = self.conv3(x, edge_index)

        #x = self.conv3(x, edge_index)
        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.num_layers == 4:
            if mode == 'visualize':
                return x
            elif mode == 'train':
                x = self.conv4(x, edge_index)
                return x
        
        if self.use_lnorm and self.num_layers != 4:
            x = self.lnorm4(self.conv4(x, edge_index))
        elif not self.use_lnorm or self.num_layers == 4:
            x = self.conv4(x, edge_index)

        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.num_layers == 5:
            if mode == 'visualize':
                return x
            elif mode == 'train':
                x = self.conv5(x, edge_index)
                return x
        
        if self.use_lnorm and self.num_layers != 5:
            x = self.lnorm5(self.conv5(x, edge_index))
        elif not self.use_lnorm or self.num_layers == 5:
            x = self.conv5(x, edge_index)
        
        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.num_layers == 6:
            if mode == 'visualize':
                return x
            elif mode == 'train':
                x = self.conv6(x, edge_index)
                return x

class GCN_basic_projection(nn.Module):
    def __init__(self, num_layers,activation,dropout,hidden_channels,input_size, output_size,use_lnorm = True):
        super().__init__()
        torch.manual_seed(1234567)
        self.use_lnorm = use_lnorm
        self.activation = activation
        self.num_layers = num_layers
        self.dropout = dropout

        "Graph Encoder"
        self.conv1 = GCNConv(input_size, hidden_channels)
        self.lnorm1 = LayerNorm(hidden_channels)
        if num_layers > 1:
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.lnorm2 = LayerNorm(hidden_channels)
            if num_layers > 2:
                self.conv3 = GCNConv(hidden_channels, hidden_channels)
                self.lnorm3 = LayerNorm(hidden_channels)
                if num_layers > 3:
                    self.conv4 = GCNConv(hidden_channels, hidden_channels)
                    self.lnorm4 = LayerNorm(hidden_channels)
                    if num_layers > 4:
                        self.conv5 = GCNConv(hidden_channels,hidden_channels)
                        self.lnorm5 = LayerNorm(hidden_channels)
                        if num_layers > 5:
                            self.conv6 = GCNConv(hidden_channels, hidden_channels)
                            self.lnorm6 = LayerNorm(hidden_channels)
                        
        "Projector"
        "For task == 'semi_sup'"
        self.project_semi_sup = Linear(hidden_channels, output_size)
        "For task == 'self_sup'"
        self.project_self_sup = Linear(hidden_channels, hidden_channels)



    def forward(self, input, edge_index, mode='train',task = 'self_sup'):

        "Layer 1"
        if self.use_lnorm:
            x = self.lnorm1(self.conv1(input, edge_index))
        else:       
            x = self.conv1(input, edge_index)
        
        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.num_layers == 1:
            if mode == 'visualize':
                return x
            elif mode == 'train':
                if task == 'classif':
                    x = self.project_semi_sup(x)
                elif task == 'self_sup':
                    x = self.project_self_sup(x)
                return x        

        "Layer 2"
        if self.use_lnorm:
            x = self.lnorm2(self.conv2(x, edge_index))
        else:
            x = self.conv2(x, edge_index)
        
        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.num_layers == 2:
            if mode == 'visualize':
                return x
            elif mode == 'train':
                if task == 'classif':
                    x = self.project_semi_sup(x)
                elif task == 'self_sup':
                    x = self.project_self_sup(x)
                return x
        
        "Layer 3"        
        if self.use_lnorm:
            x = self.lnorm3(self.conv3(x, edge_index))
        else:
            x = self.conv3(x, edge_index)

        #x = self.conv3(x, edge_index)
        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.num_layers == 3:
            if mode == 'visualize':
                return x
            elif mode == 'train':
                if task == 'classif':
                    x = self.project_semi_sup(x)
                elif task == 'self_sup':
                    x = self.project_self_sup(x)
                return x
        
        "Layer 4"        
        if self.use_lnorm:
            x = self.lnorm4(self.conv4(x, edge_index))
        else:
            x = self.conv4(x, edge_index)

        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.num_layers == 4:
            if mode == 'visualize':
                return x
            elif mode == 'train':
                if task == 'classif':
                    x = self.project_semi_sup(x)
                elif task == 'self_sup':
                    x = self.project_self_sup(x)
                return x
            
        "Layer 5"        
        if self.use_lnorm:
            x = self.lnorm5(self.conv5(x, edge_index))
        else:
            x = self.conv5(x, edge_index)
        
        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.num_layers == 5:
            if mode == 'visualize':
                return x
            elif mode == 'train':
                if task == 'classif':
                    x = self.project_semi_sup(x)
                elif task == 'self_sup':
                    x = self.project_self_sup(x)
                return x

        "Layer 6"      
        if self.use_lnorm:
            x = self.lnorm6(self.conv6(x, edge_index))
        else:
            x = self.conv6(x, edge_index)
        
        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.num_layers == 6:
            if mode == 'visualize':
                return x
            elif mode == 'train':
                if task == 'classif':
                    x = self.project_semi_sup(x)
                elif task == 'self_sup':
                    x = self.project_self_sup(x)
                return x

class SAGE_basic(nn.Module):
    def __init__(self, num_layers,activation,dropout,aggregation,hidden_channels,input_size, output_size,use_lnorm = True):
        super().__init__()
        torch.manual_seed(1234567)
        self.num_layers = num_layers
        self.use_lnorm = use_lnorm
        self.activation = activation
        self.dropout = dropout
        self.aggregation = aggregation
        if num_layers == 1:
            self.conv1 = SAGEConv(input_size, output_size,aggr=aggregation)
        else:
            self.conv1 = SAGEConv(input_size, hidden_channels,aggr=aggregation)
            self.lnorm1 = LayerNorm(hidden_channels)

        if num_layers == 6:
            self.conv2 = SAGEConv(hidden_channels, hidden_channels,aggr=aggregation)
            self.lnorm2 = LayerNorm(hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, hidden_channels,aggr=aggregation)
            self.lnorm2 = LayerNorm(hidden_channels)
            self.conv4 = SAGEConv(hidden_channels, hidden_channels,aggr=aggregation)
            self.lnorm4 = LayerNorm(hidden_channels)
            self.conv5 = SAGEConv(hidden_channels,hidden_channels,aggr=aggregation)
            self.lnorm5 = LayerNorm(hidden_channels)
            self.conv6 = SAGEConv(hidden_channels, output_size,aggr=aggregation)
        elif num_layers == 5:
            self.conv2 = SAGEConv(hidden_channels, hidden_channels,aggr=aggregation)
            self.lnorm2 = LayerNorm(hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, hidden_channels,aggr=aggregation)
            self.lnorm3 = LayerNorm(hidden_channels)
            self.conv4 = SAGEConv(hidden_channels, hidden_channels,aggr=aggregation)
            self.lnorm4 = LayerNorm(hidden_channels)
            self.conv5 = SAGEConv(hidden_channels, output_size,aggr=aggregation)
        elif num_layers == 4:
            self.conv2 = SAGEConv(hidden_channels, hidden_channels,aggr=aggregation)
            self.lnorm2 = LayerNorm(hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, hidden_channels,aggr=aggregation)
            self.lnorm3 = LayerNorm(hidden_channels)
            self.conv4 = SAGEConv(hidden_channels, output_size,aggr=aggregation)
        elif num_layers == 3:
            self.conv2 = SAGEConv(hidden_channels, hidden_channels,aggr=aggregation)
            self.lnorm2 = LayerNorm(hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, output_size,aggr=aggregation)
        elif num_layers == 2:
            self.conv2 = SAGEConv(hidden_channels, output_size,aggr=aggregation)
        

    def forward(self, input, edge_index, mode = 'train'):
        if self.use_lnorm and self.num_layers != 1:
            x = self.lnorm1(self.conv1(input, edge_index))
        else:
            x = self.conv1(input, edge_index)

        if self.num_layers == 1:
            if mode == 'visualize':
                return input
            elif mode == 'train':
                return x
        
        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.num_layers == 2:
            if mode == 'visualize':
                return x
            elif mode == 'train':
                x = self.conv2(x, edge_index)
                return x
        
        if self.use_lnorm and self.num_layers != 2:
            x = self.lnorm2(self.conv2(x, edge_index))
        else:
            x = self.conv2(x, edge_index)


        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.num_layers == 3:
            if mode == 'visualize':
                return x
            elif mode == 'train':
                x = self.conv3(x, edge_index)
                return x
        
        if self.use_lnorm and self.num_layers != 3:
            x = self.lnorm3(self.conv3(x, edge_index))
        else:
            x = self.conv3(x, edge_index)

        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.num_layers == 4:
            if mode == 'visualize':
                return x
            elif mode == 'train':
                x = self.conv4(x, edge_index)
                return x
            
        if self.use_lnorm and self.num_layers != 4:
            x = self.lnorm4(self.conv4(x, edge_index))
        else:
            x = self.conv4(x, edge_index)

        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.num_layers == 5:
            if mode == 'visualize':
                return x
            elif mode == 'train':
                x = self.conv5(x, edge_index)
                return x
        
        if self.use_lnorm and self.num_layers != 5:
            x = self.lnorm5(self.conv5(x, edge_index))
        else:
            x = self.conv5(x, edge_index)

        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.num_layers == 6:
            if mode == 'visualize':
                return x
            elif mode == 'train':
                x = self.conv6(x, edge_index)
                return x


class GAT_basic(nn.Module):
    def __init__(self, num_layers,activation,dropout,hidden_channels, input_size, output_size, heads,use_lnorm = True):
        super().__init__()
        torch.manual_seed(1234567)
        if isinstance(heads, str):
            heads = int(heads)

        self.use_lnorm = use_lnorm
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        if num_layers == 1:
            self.conv1 = GATConv(input_size,output_size, heads = heads, concat=False)  
        else:
            self.conv1 = GATConv(input_size,hidden_channels, heads = heads, concat=False)  
            self.lnorm1 = LayerNorm(hidden_channels)

        if num_layers == 6:
            self.conv2 = GATConv(hidden_channels,hidden_channels, heads = heads, concat=False)  
            self.lnorm2 = LayerNorm(hidden_channels)
            self.conv3 = GATConv(hidden_channels, hidden_channels, heads = heads, concat=False)  
            self.lnorm3 = LayerNorm(hidden_channels)
            self.conv4 = GATConv(hidden_channels, hidden_channels, heads = heads, concat=False)  
            self.lnorm4 = LayerNorm(hidden_channels)
            self.conv5 = GATConv(hidden_channels,hidden_channels, heads = heads, concat=False)  
            self.lnorm5 = LayerNorm(hidden_channels)
            self.conv6 = GATConv(hidden_channels, output_size, heads = heads, concat=False)  
        elif num_layers == 5:
            self.conv2 = GATConv(hidden_channels, hidden_channels, heads = heads, concat=False)  
            self.lnorm2 = LayerNorm(hidden_channels)
            self.conv3 = GATConv(hidden_channels, hidden_channels, heads = heads, concat=False)  
            self.lnorm3 = LayerNorm(hidden_channels)
            self.conv4 = GATConv(hidden_channels, hidden_channels, heads = heads, concat=False) 
            self.lnorm4 = LayerNorm(hidden_channels) 
            self.conv5 = GATConv(hidden_channels, output_size, heads = heads, concat=False)  
        elif num_layers == 4:
            self.conv2 = GATConv(hidden_channels, hidden_channels, heads = heads, concat=False)  
            self.lnorm2 = LayerNorm(hidden_channels)
            self.conv3 = GATConv(hidden_channels, hidden_channels, heads = heads, concat=False) 
            self.lnorm3 = LayerNorm(hidden_channels)
            self.conv4 = GATConv(hidden_channels, output_size, heads = heads, concat=False)  
        elif num_layers == 3:
            self.conv2 = GATConv(hidden_channels, hidden_channels, heads = heads, concat=False)  
            self.lnorm2 = LayerNorm(hidden_channels)
            self.conv3 = GATConv(hidden_channels, output_size, heads = heads, concat=False)  
        elif num_layers == 2:
            self.conv2 = GATConv(hidden_channels, output_size, heads = heads, concat=False)  

    def forward(self, input, edge_index,mode = 'train'):
        if self.use_lnorm and self.num_layers != 1:
            x = self.lnorm1(self.conv1(input, edge_index))
        elif not self.use_lnorm or self.num_layers == 1:
            x = self.conv1(input, edge_index)
        
        if self.num_layers == 1:
            if mode == 'visualize':
                return input
            elif mode == 'train':
                return x
        
        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        elif self.activation == 'elu':
            x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.num_layers == 2:
            if mode == 'visualize':
                return x
            elif mode == 'train':
                x = self.conv2(x, edge_index)
                return x
            
        if self.use_lnorm and self.num_layers != 2:
            x = self.lnorm2(self.conv2(x, edge_index))
        elif not self.use_lnorm or self.num_layers == 2:
            x = self.conv2(x, edge_index)

        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        elif self.activation == 'elu':
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.num_layers == 3:
            if mode == 'visualize':
                return x
            elif mode == 'train':
                x = self.conv3(x, edge_index)
                return x
            
        if self.use_lnorm and self.num_layers != 3:
            x = self.lnorm3(self.conv3(x, edge_index))
        else:
            x = self.conv3(x, edge_index)

        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        elif self.activation == 'elu':
            x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.num_layers == 4:
            if mode == 'visualize':
                return x
            elif mode == 'train':
                x = self.conv4(x, edge_index)
                return x
            
        if self.use_lnorm and self.num_layers != 4:
            x = self.lnorm4(self.conv4(x, edge_index))
        else:
            x = self.conv4(x, edge_index)

        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        elif self.activation == 'elu':
            x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.num_layers == 5:
            if mode == 'visualize':
                return x
            elif mode == 'train':
                x = self.conv5(x, edge_index)
                return x
        
        if self.use_lnorm and self.num_layers != 5:
            x = self.lnorm5(self.conv5(x, edge_index))
        else:
            x = self.conv5(x, edge_index)
        
        if self.activation == 'relu':
            x = x.relu()
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'tanh':
            x = F.tanh(x)
        elif self.activation == 'elu':
            x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.num_layers == 6:
            if mode == 'visualize':
                return x
            elif mode == 'train':
                x = self.conv6(x, edge_index)
                return x
    

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.n_layers = n_layers
        self.head_hidden_dim = hidden_dim

        a = nn.ReLU()
        for i in range(n_layers):
            start_dim = hidden_dim if i else input_dim
            conv = GCNConv(start_dim, hidden_dim)
            self.convs.append(conv)
            self.acts.append(a)

        self.proj1 = Linear(hidden_dim, hidden_dim)
        self.proj2 = Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index): #data
        #GNN encoder
        #x, edge_index, batch = data
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            x = self.acts[i](x)
        #MLP head
        x = self.proj1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.proj2(x)
        return x