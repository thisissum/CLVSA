import torch
from torch import nn
import torch.nn.functional as F


class Mask(nn.Module):
    """mask layer, output the (batch_size, seq_len, 1) mask matrix
    """

    def __init__(self, device):
        super(Mask, self).__init__()
        self.torch = torch if device == 'cpu' else torch.cuda

    def forward(self, inputs):
        output = (inputs != 0).astype(self.torch.FloatTensor)
        return output.unsqueeze(2)


class ScaleDotAttention(nn.Module):

    def __init__(self):
        super(ScaleDotAttention, self).__init__()

    def forward(self, q, k, v, mask):
        """
        q,k,v: batch_size * seq_len * feature_dim
        """
        batch_size, seq_len, dk = q.shape

        # batch_size * seq_len * seq_len
        att = torch.matmul(q, k.permute([0, 2, 1])) / float(dk**0.5)
        # add softmax to attention matrix
        att = F.softmax(att, dim=2)

        # dot product with v
        output = torch.matmul(att, v)
        return output * mask


class MultiHeadAttention(nn.Module):
    """Implementation of multi-head self attention from 'attention is all you need'
    args:
        feature_dim: int, from (batch_size, seq_len, feature_dim)
        head_num: int, num of multi head
    """

    def __init__(self, feature_dim, head_num=4):
        super(MultiHeadAttention, self).__init__()

        self.head_num = head_num
        self.feature_dim = feature_dim
        self.dim_per_head = int(feature_dim / head_num)
        for i in range(head_num):
            setattr(self, 'wq'+str(i), nn.Linear(feature_dim, self.dim_per_head))
            setattr(self, 'wk'+str(i), nn.Linear(feature_dim, self.dim_per_head))
            setattr(self, 'wv'+str(i), nn.Linear(feature_dim, self.dim_per_head))

        self.dot_attention_layer = ScaleDotAttention()

    def forward(self, q, k, v, mask):

        sub_q = []
        sub_k = []
        sub_v = []

        # use linear projection, compose the feature_dim to dim_per_head
        for i in range(self.head_num):
            sub_q.append(getattr(self, 'wq'+str(i))(q))
            sub_k.append(getattr(self, 'wk'+str(i))(k))
            sub_v.append(getattr(self, 'wv'+str(i))(v))

        # do self-attention separately
        sub_part = []
        for j in range(self.head_num):
            att = self.dot_attention_layer(sub_q[j], sub_k[j], sub_v[j], mask)
            sub_part.append(att)

        # concate heads together
        output = torch.cat(sub_part, dim=2)
        return output


class CapsuleLayer(nn.Module):
    """Implementation of capsule network (shared weight) from Hinton's papaer for sequences.
    args:
        in_num: int, input capsule num, seq_len in nlp
        in_dim: int, input capsule dim, feature_dim in nlp
        out_num: int, output capsule num, output seq_len in nlp
        out_dim: int, output capsule dim
        iter_num: int, num of iteration when doing dynamic routing
        share_weight: bool, if True, each input capsule will be conv by same matrix.
    """

    def __init__(self, in_num,
                 in_dim,
                 out_num,
                 out_dim,
                 iter_num=3,
                 share_weight=True):
        super(CapsuleLayer, self).__init__()
        self.in_num = in_num
        self.in_dim = in_dim
        self.out_num = out_num
        self.out_dim = out_dim
        self.iter_num = iter_num
        self.share_weight = share_weight
        if share_weight:
            self.W = nn.Parameter(torch.randn(out_dim*out_num, in_dim, 1))
        else:
            self.W = nn.Parameter(torch.randn(in_num, in_dim, out_dim*out_num))

    def _squash(self, x, dim=-1, constant=1):
        vec_norm = torch.pow(x, 2).sum(dim=dim, keepdim=True)
        output = torch.sqrt(vec_norm) / (constant + vec_norm) * x
        return output

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape
        if self.share_weight:
            # use conv1d to rotate and change raw vectors,
            # each timestep vec * shared weight for all vec -> new_vec
            #u_hat.shape = (batch_size, seq_len, out_dim*out_num)
            u_hat = F.conv1d(x.permute(0, 2, 1), self.W).permute(0, 2, 1)
        else:
            # each timestep vec * weight for that vec -> new_vec
            #u_hat.shape = (batch_size, seq_len, out_dim*out_num)
            u_hat = torch.matmul(x.unsqueeze(2), self.W).squeeze(2)

        # each input_capsule is mapped by a (out_num,out_dim) matrix
        u_hat = u_hat.view(batch_size, -1, self.out_num, self.out_dim)
        b = torch.zeros(batch_size, self.out_num, u_hat.shape[1])

        for r in range(self.iter_num):
            c_r = F.softmax(b, dim=1)  # u_i allocate itself by this weight
            # each weight c_ij is timed by the i_th input_capsule's (j, out_dim) vec.
            s_r = (u_hat * c_r).sum(dim=1)
            a_r = self._squash(s_r)
            if r < self.iter_num - 1:
                b = b + (a_r.unsqueeze(1) * u_hat).sum(dim=-1).permute(0, 2, 1)
        v = a_r
        return v


class LocalInferenceLayer(nn.Module):
    """Local inference modeling part of ESIM
    """

    def __init__(self):
        super(LocalInferenceLayer, self).__init__()

    def forward(self, seq_1, seq_2):
        #seq_1.shape = batch_size, seq_1_len, feature_dim
        #seq_2.shape = batch_size, seq_2_len, feature_dim

        # batch_size, seq_1_len, seq_2_len
        e_ij = torch.matmul(seq_1, seq_2.permute(0, 2, 1))

        # weighted for inference
        weighted_seq2 = F.softmax(e_ij, dim=2)
        weighted_seq1 = F.softmax(e_ij, dim=1)

        # inference
        seq_1_hat = torch.matmul(weighted_seq2, seq_2)  # same shape as seq_1
        seq_2_hat = torch.matmul(weighted_seq1.permute(0, 2, 1), seq_1)

        return seq_1_hat, seq_2_hat, seq_1_hat-seq_2_hat, seq_1_hat*seq_2_hat


class AdaptiveConv1d(nn.Module):
    """adaptive conv1d, whose filters are determained by the inputs
    args:
        feature_dim: int, embedding dim or cnn/rnn/transformer output dim
        gru_hidden_size: int, hidden_size of gru layer
        kernel_size: int, kernel_size of conv
        filter_num: int, same as filter_num
        method: str, 'full' or 'hash', method to generate filters, 'hash' for memory saving
    """

    def __init__(self, feature_dim,
                 gru_hidden_size=128,
                 kernel_size=2,
                 filter_num=100,
                 method='full'):
        super(AdaptiveConv1d, self).__init__()
        self.feature_dim = feature_dim
        self.kernel_size = kernel_size
        self.filter_num = filter_num
        self.method = method
        self.gru_layer = nn.GRU(
            input_size=feature_dim, hidden_size=gru_hidden_size,
            num_layers=1, batch_first=True, bidirectional=True)
        self.q_vector = nn.Parameter(torch.rand(gru_hidden_size*2, 1))
        self.q_vector.requires_grad = True
        if method == 'full':
            mat = torch.Tensor(filter_num, feature_dim,
                               kernel_size, gru_hidden_size*2)
            self.generation_weight = nn.Parameter(nn.init.orthogonal_(mat))
            self.generation_weight.requires_grad = True
        if method == 'hash':
            pass

    def forward(self, x):
        batch_size, seq_len, dk = x.shape
        gru_output, _ = self.gru_layer(x)
        # batch_size, seq_len, 1
        weight = F.softmax(torch.matmul(gru_output, self.q_vector), dim=1)
        # batch_size, hidden_dim*2
        context_vector = torch.sum(gru_output * weight, dim=1)
        filters_batch = torch.matmul(self.generation_weight, context_vector.permute(
            1, 0))  # filter_num, feature_dim, kernel_size, batch_size
        batch = []
        for i in range(batch_size):
            conv_out = F.relu(F.conv1d(x[i, :, :].unsqueeze(0).permute(
                0, 2, 1), weight=filters_batch[:, :, :, i]))
            batch.append(conv_out.permute(0, 2, 1))
        output = torch.cat(batch, dim=0)
        return output


class SamePaddingConv1d(nn.Module):
    """Pytorch Conv1d but padding mode is same as tensorflow's conv1d with padding='same'

    """

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 bias=True):
        super(SamePaddingConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.dilation = dilation
        self.bias = bias
        self.conv_kernel = nn.Parameter(torch.randn(
            out_channels, in_channels, kernel_size))
        if bias:
            self.bias_weight = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        batch_size, dk, seq_len = x.shape
        out_rows = (seq_len + self.stride - 1) // self.stride
        padding_rows = max(0, (out_rows-1)*self.stride +
                           (self.kernel_size-1)*self.dilation+1-seq_len)
        dilation_odd = (self.dilation % 2 != 0)
        if dilation_odd:
            if self.kernel_size % 2 != 0:
                pass
            else:
                device = x.device
                x = torch.cat([torch.zeros(batch_size, dk, 1).to(
                    device), x], axis=2).contiguous()
        if self.bias:
            output = F.conv1d(x, self.conv_kernel, bias=self.bias_weight,
                              stride=self.stride, dilation=self.dilation, padding=padding_rows//2)
        else:
            output = F.conv1d(x, self.conv_kernel, stride=self.stride,
                              dilation=self.dilation, padding=padding_rows//2)

        return output


class DGCNN(nn.Module):
    """DGCNN implement in pytorh followed Jianlin Su' idea

    """

    def __init__(self, kernel_size,
                 dilation_rate,
                 in_channels,
                 out_channels=None,
                 skip_connect=True,
                 drop_gate=None):
        super(DGCNN, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.skip_connect = skip_connect
        self.drop_gate = drop_gate

        if out_channels is None:
            self.out_channels = in_channels
        else:
            self.out_channels = out_channels

        self.conv1d_layer = SamePaddingConv1d(
            in_channels, out_channels*2, kernel_size, dilation=dilation_rate)

        if out_channels is not None and out_channels != in_channels:
            self.conv1d_1x1 = SamePaddingConv1d(
                in_channels, out_channels, kernel_size=1)

        if drop_gate is not None:
            self.dropout_layer = nn.Dropout(drop_gate)

    def forward(self, x0, mask):
        x = x0 * mask
        x = self.conv1d_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
        x, g = x[:, :, :self.out_channels], x[:, :, self.out_channels:]
        if self.drop_gate is not None:
            g = self.dropout_layer(g)
        g = F.sigmoid(g)

        if self.skip_connect:
            if self.out_channels != self.in_channels:
                x0 = self.conv1d_1x1(x0.permute(0, 2, 1)).permute(0, 2, 1)
            return (x0 * (1-g) + x * g) * mask
        else:
            return x * g * mask


"""F.conv1d()
inputs: batch_size, feature_dim, seq_len
weight: output_feature_dim, feature_dim, kernel_size
output: batch_size, out_feature_dim, new_seq_len
"""
