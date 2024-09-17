import torch
from torch.nn import LayerNorm, MultiheadAttention, Sequential, Linear, BatchNorm1d
import torch.nn.functional as F

import torch_geometric
from torch_geometric.nn import GINConv, MessagePassing, BatchNorm, InstanceNorm, GraphNorm, SAGEConv, GCNConv, GraphConv, ChebConv, GENConv, GMMConv, LEConv, SGConv, TAGConv, TransformerConv, SplineConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv

class SimpleRGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, head_cnt, dropout):
        super(SimpleRGATConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.gat_conv1 = GATv2Conv(in_channels, int(out_channels // head_cnt),
                                dropout=dropout, heads=head_cnt, bias=True, 
                                share_weights=False, add_self_loops=False)
        self.gat_conv2 = GATv2Conv(in_channels, int(out_channels // head_cnt),
                                dropout=dropout, heads=head_cnt, bias=True, 
                                share_weights=False, add_self_loops=False)
        self.gat_conv3 = GATv2Conv(in_channels, int(out_channels // head_cnt),
                                dropout=dropout, heads=head_cnt, bias=True, 
                                share_weights=False, add_self_loops=False)


    def forward(self, x, edge_index, edge_type):
        return self.gat_conv1(x, edge_index[:,edge_type == 0]) + \
                self.gat_conv2(x, edge_index[:,edge_type == 1]) + \
                self.gat_conv3(x, edge_index[:,edge_type == 2])
        

class RGINConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, aggr='add', **kwargs):
        super(RGINConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        # MLP for each relation
        self.mlps = torch.nn.ModuleList()
        for _ in range(num_relations):
            mlp = torch.nn.Sequential(torch.nn.Linear(in_channels, out_channels),
                        torch.nn.GELU(),
                        torch.nn.Linear(out_channels, out_channels),
                        torch.nn.GELU())
            self.mlps.append(mlp)

        # Learnable parameter for GIN
        self.eps = torch.nn.Parameter(torch.zeros(num_relations))

    def forward(self, x, edge_index, edge_type):
        out = 0

        loop_processed = False
        for rel in range(self.num_relations):
            try:
              mask = edge_type == rel

              select_edge_index = edge_index[:, mask]
              if select_edge_index.size(1) == 0 or x.size(0) <= 1:
                continue

              out += self.propagate(select_edge_index, x=x, edge_type=rel)
              loop_processed = True
            except Exception as e:
              if "CUDA out of memory" in str(e):
                raise e
              else:
                print(select_edge_index.shape)
                continue

        if not loop_processed:
            out = x
        return out

    def message(self, x_j, edge_type):
        # Apply corresponding MLP
        return self.mlps[edge_type](x_j) * (1 + self.eps[edge_type])

    def update(self, aggr_out, x):
        # Simply add the aggregated value with the node's own value
        return aggr_out + x

    def reset_parameters(self):
        for mlp in self.mlps:
            for layer in mlp:
                if isinstance(layer, (Linear, BatchNorm1d)):
                    layer.reset_parameters()
        self.eps.data.fill_(0)


class DTBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(DTBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_cnt = 8

        self.norm1 = LayerNorm(in_channels)

        self.patch_dim = 16

        self.qkv_proj = torch.nn.Linear(self.patch_dim, 3 * self.patch_dim)

        self.mha = MultiheadAttention(self.patch_dim, self.head_cnt, dropout=dropout, batch_first=True)
            
        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_channels, in_channels), # 2 * in_channels
                                            torch.nn.GELU(),
                                            torch.nn.Linear(in_channels, in_channels),
                                            torch.nn.GELU())

    def forward(self, x, edge_index, edge_attr):
        x1 = self.norm1(x)
        x1_new_shape = x1.reshape(x1.shape[0], x1.shape[1] // self.patch_dim, self.patch_dim)

        qkv = self.qkv_proj(x1_new_shape)
        q, k, v = qkv.chunk(3, dim=-1)
        att_out, _ = self.mha(q, k, v)
        att_out_new_shape = att_out.reshape(x1.shape[0], x1.shape[1])
        return x + att_out_new_shape + self.mlp(x1)


class GTBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout, head_cnt):
        super(GTBlock, self).__init__()

        self.in_channels = in_channels
        self.output_channels = out_channels

        self.norm1 = LayerNorm(in_channels)
        self.mha = SimpleRGATConv(in_channels, out_channels, head_cnt, dropout)

        if self.in_channels == self.output_channels:
            self.mlp = torch.nn.Sequential(torch.nn.Linear(out_channels, out_channels),
                                            torch.nn.GELU(),
                                            torch.nn.Linear(out_channels, out_channels),
                                            torch.nn.GELU())
        else:
            self.conv1 = RGINConv(in_channels, out_channels, 3)

            self.conv_norm2 = LayerNorm(out_channels)
            self.conv2 = RGINConv(out_channels, out_channels, 3)
            
            self.conv_norm3 = LayerNorm(out_channels)
            self.conv3 = RGINConv(out_channels, out_channels, 3)
            
            self.conv_norm4 = LayerNorm(out_channels)
            self.conv4 = RGINConv(out_channels, out_channels, 3)

    def forward(self, x, edge_index, edge_attr):        
        edge_type = (edge_attr + 1).int().flatten()
        x1 = self.norm1(x)
        if self.in_channels == self.output_channels:
            return x + self.mha(x1, edge_index, edge_type) + self.mlp(x1)
        else:
            
            x2 = self.conv1(x1, edge_index, edge_type)

            x2n = self.conv_norm2(x2)
            x3 = x2 + self.conv2(x2n, edge_index, edge_type)

            x3n = self.conv_norm3(x3)
            x4 = x3 + self.conv3(x3n, edge_index, edge_type)

            x4n = self.conv_norm4(x4)
            x5 = x4 + self.conv4(x4n, edge_index, edge_type)
            
            return x5


class GTModel(torch.nn.Module):
    def __init__(self, rb_num, decode_num):
        super(GTModel, self).__init__()
        assert(rb_num > 0)

        dropout = 0.2
        out_channels = 48
        head_cnt = 8

        assert(out_channels % head_cnt == 0)

        tmp_rb = []
        tmp_rb.append(self.build_layer(1, out_channels, dropout, head_cnt))

        for _ in range(rb_num - 1):
            tmp_rb.append(self.build_layer(out_channels, out_channels, dropout, head_cnt))

        self.rb = torch.nn.ModuleList(tmp_rb)

        self.decode = None
        if decode_num > 0:
            tmp_decode = []
            for _ in range(decode_num):
                tmp_decode.append(self.decode_layer(out_channels, out_channels, dropout))

            self.decode = torch.nn.ModuleList(tmp_decode)
        
        self.mlp1 = torch.nn.Linear(out_channels, out_channels // 2)
        self.mlp2 = torch.nn.Linear(out_channels // 2, out_channels // 4)
        self.mlp3 = torch.nn.Linear(out_channels // 4, 1)

        
    def build_layer(self, input_dim, output_dim, dropout, head_cnt):
        return GTBlock(input_dim, output_dim, dropout, head_cnt)

    def decode_layer(self, input_dim, output_dim, dropout):
        return DTBlock(input_dim, output_dim, dropout)
        
    def forward(self, x, edge_index, edge_attr):
        for i in range(0, len(self.rb)):
            x = self.rb[i](x, edge_index, edge_attr)

        if self.decode != None:
            for i in range(0, len(self.decode)):
                x = self.decode[i](x, edge_index, edge_attr)

        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        x = F.gelu(x)
        x = self.mlp3(x)
        return F.sigmoid(x)