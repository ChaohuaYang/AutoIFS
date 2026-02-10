import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import MultiLayerPerceptron, FeatureEmbedding
import modules.layers as layer


class BasicModel(torch.nn.Module):
    def __init__(self, opt):
        super(BasicModel, self).__init__()
        self.latent_dim = opt["latent_dim"]
        self.feature_num = opt["feat_num"]
        self.field_num = opt["field_num"]
        self.task_num = opt['task_num']
        self.domain_num = len(opt['domain_list'])
        self.lora_r = opt['lora_r']
        self.ticket = False
        print(self.field_num)
        print(self.feature_num)
        self.embedding = FeatureEmbedding(self.feature_num, self.latent_dim)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, field_num)``

        """
        pass

    def reg(self):
        return 0.0


class AutoIFS(BasicModel):
    def __init__(self, opt):
        super(AutoIFS, self).__init__(opt)
        embed_dims = opt["mlp_dims"]
        dropout = opt["mlp_dropout"]
        self.use_bn = opt["use_bn"]
        self.dnn_dim = self.field_num * self.latent_dim
        self.temp = 1
        self.share_units = embed_dims[:2]
        self.hidden_units = embed_dims[2:] + [1]
        # print(self.hidden_units)
        self.hidden_units_d = self.hidden_units[:1]
        self.hidden_units_t = self.hidden_units[1:]
        hidden_units_d = [self.share_units[-1]] + self.hidden_units_d
        hidden_units_t = [self.hidden_units_d[-1]] + self.hidden_units_t
        self.share_mlp = MultiLayerPerceptron(self.dnn_dim, self.share_units, output_layer=False, dropout=dropout,
                                              use_bn=self.use_bn)
        self.D_kernels = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_units_d[i], hidden_units_d[i + 1]))
            for i in range(len(self.hidden_units_d))
        ])
        self.D_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(self.hidden_units_d[i]))
            for i in range(len(self.hidden_units_d))
        ])

        self.T_kernels = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_units_t[i], hidden_units_t[i + 1]))
            for i in range(len(self.hidden_units_t))
        ])
        self.T_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(self.hidden_units_t[i]))
            for i in range(len(self.hidden_units_t))
        ])

        self.D_A_kernel = nn.ParameterList([
            nn.Parameter(torch.empty(self.domain_num, hidden_units_d[i], self.lora_r))
            for i in range(len(self.hidden_units_d))
        ])
        self.D_B_kernel = nn.ParameterList([
            nn.Parameter(torch.zeros(self.domain_num, self.lora_r, hidden_units_d[i + 1]))
            for i in range(len(self.hidden_units_d))
        ])
        self.D_lora_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(self.domain_num, self.hidden_units_d[i]))
            for i in range(len(self.hidden_units_d))
        ])

        self.T_A_kernel = nn.ParameterList([
            nn.Parameter(torch.empty(self.task_num, hidden_units_t[i], self.lora_r))
            for i in range(len(self.hidden_units_t))
        ])
        self.T_B_kernel = nn.ParameterList([
            nn.Parameter(torch.zeros(self.task_num, self.lora_r, hidden_units_t[i + 1]))
            for i in range(len(self.hidden_units_t))
        ])
        self.T_lora_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(self.task_num, self.hidden_units_t[i]))
            for i in range(len(self.hidden_units_t))
        ])

        if self.use_bn:
            self.bn_layers = nn.ModuleList(
                [nn.BatchNorm1d(self.hidden_units[i]) for i in range(len(self.hidden_units))])

        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(self.hidden_units))])

        self.activation_layers = [nn.ReLU() for _ in range(len(self.hidden_units))]

        self.reset_parameters()

        self.gate_hypernet = MultiLayerPerceptron(self.dnn_dim, self.share_units, output_layer=False, dropout=dropout,
                                                  use_bn=self.use_bn)
        self.gate_head0 = nn.Linear(self.share_units[-1], 4)
        self.gate_head1 = nn.Linear(self.share_units[-1], 4)

        self.gate_embedding = FeatureEmbedding(self.feature_num, self.latent_dim)
        self.gate_embedding.embedding.requires_grad = False

    def reset_parameters(self):
        for kernel in self.D_kernels:
            nn.init.xavier_normal_(kernel)

        for kernel in self.T_kernels:
            nn.init.xavier_normal_(kernel)

        for a in self.D_A_kernel:
            nn.init.xavier_normal_(a)

        for a in self.T_A_kernel:
            nn.init.xavier_normal_(a)

    def forward(self, x, d):
        d = d.long()
        k = 0
        x_embedding = self.embedding(x)
        x_emb = x_embedding.view(-1, self.dnn_dim)
        x_dnn = self.share_mlp(x_emb)
        for i in range(len(self.hidden_units_d)):
            x_dnn_d_share = torch.matmul(x_dnn, self.D_kernels[i]) + self.D_bias[i]
            x_dnn_d_lora = torch.matmul(x_dnn.unsqueeze(1), self.D_A_kernel[i][d].squeeze(1))
            x_dnn_d_lora = torch.matmul(x_dnn_d_lora, self.D_B_kernel[i][d].squeeze(1)) + self.D_lora_bias[i][d]
            x_dnn = x_dnn_d_share + x_dnn_d_lora.squeeze(1)
            if k + 1 < len(self.hidden_units):
                if self.use_bn:
                    x_dnn = self.bn_layers[k](x_dnn)
                x_dnn = self.activation_layers[k](x_dnn)
                x_dnn = self.dropout_layers[k](x_dnn)
                k += 1
            if i + 1 == len(self.hidden_units_d):
                d_dnn = x_dnn
                d_share = x_dnn_d_share
                d_lora = x_dnn_d_lora.squeeze(1)

        t_dnn = [d_dnn, d_dnn]
        t_share = [d_share, d_share]
        t_lora = [d_lora, d_lora]
        t_share_lora = []
        t_lora_share = []
        for i in range(len(self.hidden_units_t)):
            for t in range(self.task_num):
                t_dnn_t_share = torch.matmul(t_dnn[t], self.T_kernels[i]) + self.T_bias[i]
                t_share_t_share = torch.matmul(t_share[t], self.T_kernels[i]) + self.T_bias[i]
                t_lora_t_share = torch.matmul(t_lora[t], self.T_kernels[i]) + self.T_bias[i]
                t_lora_share.append(t_lora_t_share)

                t_dnn_t_lora = torch.matmul(t_dnn[t], self.T_A_kernel[i][t])
                t_dnn_t_lora = torch.matmul(t_dnn_t_lora, self.T_B_kernel[i][t]) + self.T_lora_bias[i][t].unsqueeze(0)
                t_share_t_lora = torch.matmul(t_share[t], self.T_A_kernel[i][t])
                t_share_t_lora = torch.matmul(t_share_t_lora, self.T_B_kernel[i][t]) + self.T_lora_bias[i][t].unsqueeze(0)
                t_lora_t_lora = torch.matmul(t_lora[t], self.T_A_kernel[i][t])
                t_lora_t_lora = torch.matmul(t_lora_t_lora, self.T_B_kernel[i][t]) + self.T_lora_bias[i][t].unsqueeze(0)

                t_dnn[t] = t_dnn_t_share + t_dnn_t_lora
                t_share[t] = t_share_t_share
                t_lora[t] = t_lora_t_lora
                t_share_lora.append(t_share_t_lora)
                if k + 1 < len(self.hidden_units):
                    if self.use_bn:
                        t_dnn[t] = self.bn_layers[k](t_dnn[t])
                        t_share[t] = self.bn_layers[k](t_share[t])
                        t_lora[t] = self.bn_layers[k](t_lora[t])
                        t_share_lora[t] = self.bn_layers[k](t_share_lora[t])
                        t_lora_share[t] = self.bn_layers[k](t_lora_share[t])
                    t_dnn[t] = self.activation_layers[k](t_dnn[t])
                    t_dnn[t] = self.dropout_layers[k](t_dnn[t])
                    t_share[t] = self.activation_layers[k](t_share[t])
                    t_share[t] = self.dropout_layers[k](t_share[t])
                    t_lora[t] = self.activation_layers[k](t_lora[t])
                    t_lora[t] = self.dropout_layers[k](t_lora[t])
                    t_share_lora[t] = self.activation_layers[k](t_share_lora[t])
                    t_share_lora[t] = self.dropout_layers[k](t_share_lora[t])
                    t_lora_share[t] = self.activation_layers[k](t_lora_share[t])
                    t_lora_share[t] = self.dropout_layers[k](t_lora_share[t])
                    k += 1

        logit = {}
        if not self.ticket:
            hypernet_output = self.gate_hypernet(x_emb.detach())
        else:
            gate_emb = self.gate_embedding(x).view(-1, self.dnn_dim)
            hypernet_output = self.gate_hypernet(gate_emb.detach())
        gate0 = self.gate_head0(hypernet_output)
        gate1 = self.gate_head1(hypernet_output)
        self.weights = [gate0, gate1]
        for t in range(self.task_num):
            if self.ticket:
                gate = (self.weights[t] > 0.0).float()
            else:
                gate = torch.sigmoid(self.temp * self.weights[t])
            logit[t] = t_dnn[t] - gate[:, 0].unsqueeze(1) * t_share[t] - gate[:, 1].unsqueeze(1) * t_lora[t] - gate[:, 2].unsqueeze(
                1) * t_share_lora[t] - gate[:, 3].unsqueeze(1) * t_lora_share[t]
        return logit


def getOptim(network, optim, lr, l2):
    weight_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'gate' not in p[0], network.named_parameters()))
    mask_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'gate' in p[0], network.named_parameters()))

    optim = optim.lower()
    if optim == "sgd":
        return [torch.optim.SGD(weight_params, lr=lr, weight_decay=l2), torch.optim.SGD(mask_params, lr=0.01*lr)]
    elif optim == "adam":
        return [torch.optim.Adam(weight_params, lr=lr, weight_decay=l2), torch.optim.Adam(mask_params, lr=0.01*lr)]
    else:
        raise ValueError("Invalid optimizer type: {}".format(optim))