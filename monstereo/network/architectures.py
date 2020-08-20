
import torch
import torch.nn as nn


class SimpleModel(nn.Module):

    def __init__(self, input_size, output_size=2, linear_size=512, p_dropout=0.2, num_stage=3, device='cuda'):
        super(SimpleModel, self).__init__()

        self.num_stage = num_stage
        self.stereo_size = input_size
        self.mono_size = int(input_size / 2)
        self.output_size = output_size - 1
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.linear_stages = []
        self.device = device

        # Initialize weights

        # Preprocessing
        self.w1 = nn.Linear(self.stereo_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        # Internal loop
        for _ in range(num_stage):
            self.linear_stages.append(MyLinearSimple(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # Post processing
        self.w2 = nn.Linear(self.linear_size, self.linear_size)
        self.w3 = nn.Linear(self.linear_size, self.linear_size)
        self.batch_norm3 = nn.BatchNorm1d(self.linear_size)

        # ------------------------Other----------------------------------------------
        # Auxiliary
        self.w_aux = nn.Linear(self.linear_size, 1)

        # Final
        self.w_fin = nn.Linear(self.linear_size, self.output_size)

        # NO-weight operations
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):

        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        # Auxiliary task
        y = self.w2(y)
        aux = self.w_aux(y)

        # Final layers
        y = self.w3(y)
        y = self.batch_norm3(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.w_fin(y)

        # Cat with auxiliary task
        y = torch.cat((y, aux), dim=1)
        return y


class MyLinearSimple(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(MyLinearSimple, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):

        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


class DecisionModel(nn.Module):

    def __init__(self, input_size, output_size=2, linear_size=512, p_dropout=0.2, num_stage=3, device='cuda:1'):
        super(DecisionModel, self).__init__()

        self.num_stage = num_stage
        self.stereo_size = input_size
        self.mono_size = int(input_size / 2)
        self.output_size = output_size - 1
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.linear_stages_mono, self.linear_stages_stereo, self.linear_stages_dec = [], [], []
        self.device = device

        # Initialize weights

        # ------------------------Stereo----------------------------------------------
        # Preprocessing
        self.w1_stereo = nn.Linear(self.stereo_size, self.linear_size)
        self.batch_norm_stereo = nn.BatchNorm1d(self.linear_size)

        # Internal loop
        for _ in range(num_stage):
            self.linear_stages_stereo.append(MyLinear_stereo(self.linear_size, self.p_dropout))
        self.linear_stages_stereo = nn.ModuleList(self.linear_stages_stereo)

        # Post processing
        self.w2_stereo = nn.Linear(self.linear_size, self.output_size)

        # ------------------------Mono----------------------------------------------
        # Preprocessing
        self.w1_mono = nn.Linear(self.mono_size, self.linear_size)
        self.batch_norm_mono = nn.BatchNorm1d(self.linear_size)

        # Internal loop
        for _ in range(num_stage):
            self.linear_stages_mono.append(MyLinear_stereo(self.linear_size, self.p_dropout))
        self.linear_stages_mono = nn.ModuleList(self.linear_stages_mono)

        # Post processing
        self.w2_mono = nn.Linear(self.linear_size, self.output_size)

        # ------------------------Decision----------------------------------------------
        # Preprocessing
        self.w1_dec = nn.Linear(self.stereo_size, self.linear_size)
        self.batch_norm_dec = nn.BatchNorm1d(self.linear_size)
        #
        # Internal loop
        for _ in range(num_stage):
            self.linear_stages_dec.append(MyLinear(self.linear_size, self.p_dropout))
        self.linear_stages_dec = nn.ModuleList(self.linear_stages_dec)

        # Post processing
        self.w2_dec = nn.Linear(self.linear_size, 1)

        # ------------------------Other----------------------------------------------

        # NO-weight operations
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x, label=None):

        # Mono
        y_m = self.w1_mono(x[:, 0:34])
        y_m = self.batch_norm_mono(y_m)
        y_m = self.relu(y_m)
        y_m = self.dropout(y_m)

        for i in range(self.num_stage):
            y_m = self.linear_stages_mono[i](y_m)
        y_m = self.w2_mono(y_m)

        # Stereo
        y_s = self.w1_stereo(x)
        y_s = self.batch_norm_stereo(y_s)
        y_s = self.relu(y_s)
        y_s = self.dropout(y_s)

        for i in range(self.num_stage):
            y_s = self.linear_stages_stereo[i](y_s)
        y_s = self.w2_stereo(y_s)

        # Decision
        y_d = self.w1_dec(x)
        y_d = self.batch_norm_dec(y_d)
        y_d = self.relu(y_d)
        y_d = self.dropout(y_d)

        for i in range(self.num_stage):
            y_d = self.linear_stages_dec[i](y_d)
        aux = self.w2_dec(y_d)

        # Combine
        if label is not None:
            gate = label
        else:
            gate = torch.where(torch.sigmoid(aux) > 0.3,
                               torch.tensor([1.]).to(self.device), torch.tensor([0.]).to(self.device))
        y = gate * y_s + (1-gate) * y_m

        # Cat with auxiliary task
        y = torch.cat((y, aux), dim=1)
        return y


class AttentionModel(nn.Module):

    def __init__(self, input_size, output_size=2, linear_size=512, p_dropout=0.2, num_stage=3, device='cuda'):
        super(AttentionModel, self).__init__()

        self.num_stage = num_stage
        self.stereo_size = input_size
        self.mono_size = int(input_size / 2)
        self.output_size = output_size - 1
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.linear_stages_mono, self.linear_stages_stereo, self.linear_stages_comb = [], [], []
        self.device = device

        # Initialize weights
        # ------------------------Stereo----------------------------------------------
        # Preprocessing
        self.w1_stereo = nn.Linear(self.stereo_size, self.linear_size)
        self.batch_norm_stereo = nn.BatchNorm1d(self.linear_size)

        # Internal loop
        for _ in range(num_stage):
            self.linear_stages_stereo.append(MyLinear_stereo(self.linear_size, self.p_dropout))
        self.linear_stages_stereo = nn.ModuleList(self.linear_stages_stereo)

        # Post processing
        self.w2_stereo = nn.Linear(self.linear_size, self.linear_size)

        # ------------------------Mono----------------------------------------------
        # Preprocessing
        self.w1_mono = nn.Linear(self.mono_size, self.linear_size)
        self.batch_norm_mono = nn.BatchNorm1d(self.linear_size)

        # Internal loop
        for _ in range(num_stage):
            self.linear_stages_mono.append(MyLinear_stereo(self.linear_size, self.p_dropout))
        self.linear_stages_mono = nn.ModuleList(self.linear_stages_mono)

        # Post processing
        self.w2_mono = nn.Linear(self.linear_size, self.linear_size)

        # ------------------------Combined----------------------------------------------
        # Preprocessing
        self.w1_comb = nn.Linear(self.linear_size, self.linear_size)
        self.batch_norm_comb = nn.BatchNorm1d(self.linear_size)
        #
        # Internal loop
        for _ in range(num_stage):
            self.linear_stages_comb.append(MyLinear(self.linear_size, self.p_dropout))
        self.linear_stages_comb = nn.ModuleList(self.linear_stages_comb)

        # Post processing
        self.w2_comb = nn.Linear(self.linear_size, self.linear_size)

        # ------------------------Other----------------------------------------------
        # Auxiliary
        self.w_aux = nn.Linear(self.linear_size, 1)

        # Final
        self.w_fin = nn.Linear(self.linear_size, self.output_size)

        # NO-weight operations
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x, label=None):


        # Mono
        y_m = self.w1_mono(x[:, 0:34])
        y_m = self.batch_norm_mono(y_m)
        y_m = self.relu(y_m)
        y_m = self.dropout(y_m)

        for i in range(self.num_stage):
            y_m = self.linear_stages_mono[i](y_m)
        y_m = self.w2_mono(y_m)

        # Stereo
        y_s = self.w1_stereo(x)
        y_s = self.batch_norm_stereo(y_s)
        y_s = self.relu(y_s)
        y_s = self.dropout(y_s)

        for i in range(self.num_stage):
            y_s = self.linear_stages_stereo[i](y_s)
        y_s = self.w2_stereo(y_s)

        # Auxiliary task
        aux = self.w_aux(y_s)

        # Combined
        if label is not None:
            gate = label
        else:
            gate = torch.where(torch.sigmoid(aux) > 0.3,
                               torch.tensor([1.]).to(self.device), torch.tensor([0.]).to(self.device))
        y_c = gate * y_s + (1-gate) * y_m
        y_c = self.w1_comb(y_c)
        y_c = self.batch_norm_comb(y_c)
        y_c = self.relu(y_c)
        y_c = self.dropout(y_c)
        y_c = self.w_fin(y_c)

        # Cat with auxiliary task
        y = torch.cat((y_c, aux), dim=1)
        return y


class MyLinear_stereo(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(MyLinear_stereo, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        # self.w0_a = nn.Linear(self.l_size, self.l_size)
        # self.batch_norm0_a = nn.BatchNorm1d(self.l_size)
        # self.w0_b = nn.Linear(self.l_size, self.l_size)
        # self.batch_norm0_b = nn.BatchNorm1d(self.l_size)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        #
        # x = self.w0_a(x)
        # x = self.batch_norm0_a(x)
        # x = self.w0_b(x)
        # x = self.batch_norm0_b(x)

        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


class MonolocoModel(nn.Module):
    """
    Architecture inspired by https://github.com/una-dinosauria/3d-pose-baseline
    Pytorch implementation from: https://github.com/weigq/3d_pose_baseline_pytorch
    """

    def __init__(self, input_size, output_size=2, linear_size=256, p_dropout=0.2, num_stage=3):
        super(MonolocoModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for _ in range(num_stage):
            self.linear_stages.append(MyLinear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        y = self.w2(y)
        return y


class MyLinear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(MyLinear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):

        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out
