import torch 
from torch import nn 
import torch.nn.functional as F
from torch_contrib.layers import SamePaddingConv1d


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM cell for ConvLSTM, from paper CLVSA, use conv1d to capture local pattern, 
    assert batch_first==True
    """
    def __init__(self, input_shape, hidden_size, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.time_step, self.feature_num = input_shape
        self.hidden_size = hidden_size
        self.bias = bias 
        self.conv1d = SamePaddingConv1d(self.feature_num + hidden_size, hidden_size*4,kernel_size,bias=bias)
    
    def forward(self, input_tensor, hidden_state):
        #input_tensor:(batch_size, time_step, feature_num)
        batch_size, _, _ = input_tensor.shape
        h_curr, c_curr = hidden_state
        
        combine_tensor = torch.cat([input_tensor, h_curr], dim=2)

        conv_combine = self.conv1d(combine_tensor.permute(0,2,1)).permute(0,2,1)
        conv4i, conv4f, conv4o, conv4g = torch.split(conv_combine, self.hidden_size, dim=2)
        it = F.sigmoid(conv4i)
        ft = F.sigmoid(conv4f)
        ot = F.sigmoid(conv4o)
        gt = F.tanh(conv4g)

        c_next = ft * c_curr + it * gt 
        h_next = ot * F.tanh(c_next)
        return h_next, c_next 

    def init_hidden(self, batch_size, device):
        return torch.autograd.Variable(torch.zeros(batch_size, self.time_step, self.hidden_size)).to(device),\
            torch.autograd.Variable(torch.zeros(batch_size, self.time_step, self.hidden_size)).to(device)


class ConvLSTM(nn.Module):
    """Single layer ConvLSTM
    """
    def __init__(self, input_shape, hidden_size, kernel_size, bias=True):
        super(ConvLSTM, self).__init__()
        self.time_step, self.feature_num = input_shape
        self.hidden_size = hidden_size
        self.bias = bias 
        self.single_cell = ConvLSTMCell(input_shape, hidden_size, kernel_size, bias)
    
    def forward(self, input_tensor):
        batch_size, seq_len, time_step, feature_num = input_tensor.shape
        device = input_tensor.device
        #init hidden state
        h, c = self.single_cell.init_hidden(batch_size, device)
        #get output list
        output_list = []
        for t in range(seq_len):
            h, c = self.single_cell(input_tensor[:,t,:,:], (h, c))
            output_list.append(h.unsqueeze(1))
        output = torch.cat(output_list, dim=1)
        return output, (h, c)


class SelfAttention(nn.Module):
    """
    """
    def __init__(self, feature_dim, seq_len):
        super(SelfAttention, self).__init__()
        self.feature_dim =feature_dim
        self.compress = nn.Linear(2 * feature_dim, feature_dim)
        att_mask = torch.ones(seq_len,seq_len) - torch.tril(torch.ones(seq_len,seq_len))
        self.att_mask = nn.Parameter(att_mask.bool(), requires_grad=False)
    
    def forward(self, x):
        #assume x.shape=(batch_size, seq_len, feature_dim)
        #feature_dim is flattened by time_step * feature_num
        batch_size, seq_len = x.shape[0], x.shape[1]
        similar = torch.matmul(x, x.permute(0,2,1))
        #weight shape (batch_size, seq_len, seq_len)
        weight = F.softmax(similar.masked_fill(self.att_mask.expand(batch_size, seq_len, seq_len),float('-inf')), dim=2)
        after_att_out = torch.matmul(weight, x)
        output = self.compress(torch.cat([x, after_att_out], dim=2))
        return output


class InterAttention(nn.Module):
    """
    """
    def __init__(self, hidden_dim):
        super(InterAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, decoder_curr, encoder_output):
        output = F.softmax(
            (encoder_output * decoder_curr).sum(dim=2), dim=1)
        return output