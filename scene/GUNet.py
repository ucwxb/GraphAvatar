import torch
import torch.nn as nn
import math
import numpy as np
import utils.mesh_operations as mesh_operations
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.cheb_conv import ChebConv
from torch_geometric.nn.conv.gin_conv import GINConv
from torch_geometric.utils import remove_self_loops

def safe_matmul(a, b, name=""):
    # 检查输入
    if torch.isnan(a).any() or torch.isnan(b).any():
        print(f"Warning: NaN in matmul input ({name})")
        a = torch.nan_to_num(a, nan=0.0)
        b = torch.nan_to_num(b, nan=0.0)
    
    # 检查数值范围
    if torch.abs(a).max() > 1e6 or torch.abs(b).max() > 1e6:
        print(f"Warning: Large values in matmul input ({name})")
        a = torch.clamp(a, -1e6, 1e6)
        b = torch.clamp(b, -1e6, 1e6)
    
    try:
        # 使用 torch.matmul 计算
        result = torch.matmul(a, b)
        
        # 检查结果
        if torch.isnan(result).any():
            print(f"Warning: NaN in matmul output ({name})")
            result = torch.nan_to_num(result, nan=0.0)
        
        # 裁剪过大的值
        if torch.abs(result).max() > 1e6:
            print(f"Warning: Large values in matmul output ({name})")
            result = torch.clamp(result, -1e6, 1e6)
            
        return result
        
    except RuntimeError as e:
        print(f"Error in matmul ({name}): {str(e)}")
        # 如果失败，尝试在 CPU 上计算
        return torch.matmul(a.cpu(), b.cpu()).cuda()

def safe_matmul(a, b, name=""):
    # 检查输入
    if torch.isnan(a).any() or torch.isnan(b).any():
        print(f"Warning: NaN in matmul input ({name})")
        a = torch.nan_to_num(a, nan=0.0)
        b = torch.nan_to_num(b, nan=0.0)
    
    # 检查数值范围
    if torch.abs(a).max() > 1e6 or torch.abs(b).max() > 1e6:
        print(f"Warning: Large values in matmul input ({name})")
        a = torch.clamp(a, -1e6, 1e6)
        b = torch.clamp(b, -1e6, 1e6)
    
    try:
        # 使用 torch.matmul 计算
        result = torch.matmul(a, b)
        
        # 检查结果
        if torch.isnan(result).any():
            print(f"Warning: NaN in matmul output ({name})")
            result = torch.nan_to_num(result, nan=0.0)
        
        # 裁剪过大的值
        if torch.abs(result).max() > 1e6:
            print(f"Warning: Large values in matmul output ({name})")
            result = torch.clamp(result, -1e6, 1e6)
            
        return result
        
    except RuntimeError as e:
        print(f"Error in matmul ({name}): {str(e)}")
        # 如果失败，尝试在 CPU 上计算
        return torch.matmul(a.cpu(), b.cpu()).cuda()
        
    except Exception as e:
        print(f"Error in forward pass: {str(e)}")
        raise

class FrequenEncoding(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        min_freq_exp: float,
        max_freq_exp: float,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.include_input = include_input
        self.out_dim = self.get_out_dim()

    def get_out_dim(self) -> int:
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def pytorch_fwd(self,in_tensor):
        scaled_in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies).to(in_tensor.device)
        scaled_inputs = scaled_in_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

        while True:
            encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))
            if not encoded_inputs.isnan().any():
                break

        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)
        return encoded_inputs

    def forward(self, in_tensor):
        return self.pytorch_fwd(in_tensor)

class Norm(nn.Module):

    def __init__(self, norm_type, hidden_dim=64, print_info=None):
        super(Norm, self).__init__()
        self.norm = None
        self.print_info = print_info
        if norm_type == 'bn':
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == 'gn':
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, tensor):

        batch_list = [tensor.shape[1]]
        batch_size = tensor.shape[0]
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias

def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor

def normal(tensor, mean, std):
    if tensor is not None:
        torch.nn.init.normal_(tensor, mean=mean, std=std)

class GINConv_Ava(GINConv):
    def __init__(self, nn):
        super(GINConv_Ava, self).__init__(nn=nn)

    def forward(self, x, edge_index, norm):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        out = self.nn((1 + self.eps) * x + self.propagate(edge_index, x=x), norm=norm)
        return out

class ChebConv_Ava(ChebConv):
    def __init__(self, in_channels, out_channels, K, normalization=None, bias=True):
        super(ChebConv_Ava, self).__init__(in_channels, out_channels, K, normalization, bias)

    def reset_parameters(self):
        normal(self.weight, 0, 0.1)
        normal(self.bias, 0, 0.1)

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, dtype=None):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return edge_index, -deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, norm):
        try:
            Tx_0 = x
            # 使用安全的矩阵乘法
            out = safe_matmul(Tx_0, self.weight[0], "initial")

            x = x.transpose(0,1)
            Tx_0 = x
            if self.weight.size(0) > 1:
                Tx_1 = self.propagate(edge_index, x=x, norm=norm)
                Tx_1_transpose = Tx_1.transpose(0, 1)
                out = out + safe_matmul(Tx_1_transpose, self.weight[1], "first_layer")

            for k in range(2, self.weight.size(0)):
                prop_result = self.propagate(edge_index, x=Tx_1, norm=norm)
                prop_result = torch.clamp(prop_result, -1e6, 1e6)
                
                Tx_2 = 2 * prop_result - Tx_0
                Tx_2 = torch.clamp(Tx_2, -1e6, 1e6)
                
                Tx_2_transpose = Tx_2.transpose(0, 1)
                tmp = safe_matmul(Tx_2_transpose, self.weight[k], f"layer_{k}")
                
                out = out + tmp
                Tx_0, Tx_1 = Tx_1, Tx_2

            if self.bias is not None:
                out = out + self.bias
            
            if torch.isnan(out).any():
                print("Warning: NaN in final output")
                out = torch.nan_to_num(out, nan=0.0)
            
            return out
            
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            raise

    def message(self, x_j, norm):
        return norm.view(-1, 1, 1) * x_j


class Pool(MessagePassing):
    def __init__(self):
        super(Pool, self).__init__(flow='target_to_source')

    def forward(self, x, pool_mat, dtype=None):
        x = x.transpose(0,1)
        out = self.propagate(edge_index=pool_mat._indices(), x=x, norm=pool_mat._values(), size=pool_mat.size())
        return out.transpose(0,1)

    def message(self, x_j, norm):
        return norm.view(-1, 1, 1) * x_j

class GUNet(torch.nn.Module):

    def __init__(self, input_channels, output_channels, template_mesh, exp_num=0, downsampling_factors = [4, 4, 4, 4], \
                 polygon_order = [6, 6, 6, 6, 6], z = 8, num_conv_filters = [16, 16, 16, 32, 32], n_layers = 4):
        super(GUNet, self).__init__()

        self.pos_enc = FrequenEncoding(in_dim=input_channels, num_frequencies=30, \
                                    min_freq_exp=0.0, max_freq_exp=30, \
                                    include_input=True)
        input_channels  = self.pos_enc.out_dim

        D_t, U_t, A_t, num_nodes = self.initialize_param(template_mesh, downsampling_factors)
        self.graph_vertex_num = num_nodes[0]
        self.n_layers = n_layers
        self.enc_filters = num_conv_filters.copy()
        self.enc_filters.insert(0, input_channels)
        self.dec_filters = num_conv_filters.copy()
        self.dec_filters.insert(0, output_channels)
        self.K = polygon_order
        self.z = z
        self.downsample_matrices = D_t
        self.upsample_matrices = U_t
        self.adjacency_matrices = A_t

        self.A_edge_index, self.A_norm = zip(*[ChebConv_Ava.norm(self.adjacency_matrices[i]._indices(),
                                                                  num_nodes[i]) for i in range(len(num_nodes))])

        self.cheb = torch.nn.ModuleList([ChebConv_Ava(self.enc_filters[i], self.enc_filters[i+1], self.K[i])
                                         for i in range(len(self.enc_filters)-2)])
        self.cheb_dec = torch.nn.ModuleList([ChebConv_Ava(self.dec_filters[-i-1], self.dec_filters[-i-2], self.K[i])
                                             for i in range(len(self.dec_filters)-1)])

        self.pool = Pool()
        self.enc_lin = torch.nn.Linear(self.downsample_matrices[-1].shape[0]*self.enc_filters[-1], self.z)
        self.dec_lin = torch.nn.Linear(self.z + exp_num, self.dec_filters[-1]*self.upsample_matrices[-1].shape[1])

        self.act = nn.ELU()
        self.reset_parameters()

    def initialize_param(self, template_mesh, downsampling_factors = [4, 4, 4, 4]):
        M, A, D, U = mesh_operations.generate_transform_matrices(template_mesh, downsampling_factors)
        D_t = [scipy_to_torch_sparse(d).cuda() for d in D]
        U_t = [scipy_to_torch_sparse(u).cuda() for u in U]
        A_t = [scipy_to_torch_sparse(a).cuda() for a in A]
        num_nodes = [len(M[i].v) for i in range(len(M))]
        return D_t, U_t, A_t, num_nodes

    def forward(self, x, edge_index, concat_exp=False, exp=None):
        x = self.pos_enc(x)
        x = x.reshape(1, -1, self.enc_filters[0])
        x = self.encoder(x)
        if concat_exp:
            exp = exp.unsqueeze(0)
            x = torch.cat([x, exp], dim=-1)
        x = self.decoder(x)
        x = x.reshape(-1, self.dec_filters[0])
        return x

    def forward_encoder(self, x, edge_index):
        x = self.pos_enc(x)
        x = x.reshape(1, -1, self.enc_filters[0])
        x = self.encoder(x)
        return x

    def forward_decoder(self, x, edge_index, concat_exp=False, exp=None):
        if concat_exp:
            exp = exp.unsqueeze(0)
            x = torch.cat([x, exp], dim=-1)
        x = self.decoder(x)
        x = x.reshape(-1, self.dec_filters[0])
        return x

    def encoder(self, x):
        for i in range(self.n_layers):

            x = self.cheb[i](x, self.A_edge_index[i], self.A_norm[i])

            x = self.act(x)

            x = self.pool(x, self.downsample_matrices[i])

        x = x.reshape(x.shape[0], self.enc_lin.in_features)
        x = self.enc_lin(x)
        x = self.act(x)
        return x

    def decoder(self, x):
        x = self.dec_lin(x)
        x = self.act(x)
        x = x.reshape(x.shape[0], -1, self.dec_filters[-1])
        for i in range(self.n_layers):
            x = self.pool(x, self.upsample_matrices[-i-1])
            x = self.cheb_dec[i](x, self.A_edge_index[self.n_layers-i-1], self.A_norm[self.n_layers-i-1])
            x = self.act(x)
        x = self.cheb_dec[-1](x, self.A_edge_index[-1], self.A_norm[-1])
        return x

    def reset_parameters(self):
        torch.nn.init.normal_(self.enc_lin.weight, 0, 0.1)
        torch.nn.init.normal_(self.dec_lin.weight, 0, 0.1)

def safe_sin(x):
    # 检查输入是否包含 NaN
    if torch.isnan(x).any():
        print("Warning: NaN in sin input")
        x = torch.nan_to_num(x, nan=0.0)
    
    # 裁剪输入范围，避免数值溢出
    x = torch.clamp(x, -1e6, 1e6)
    
    # 计算 sin
    result = torch.sin(x)
    
    # 确保输出在合理范围内（sin 的值域是 [-1, 1]）
    result = torch.clamp(result, -1.0, 1.0)
    
    return result

