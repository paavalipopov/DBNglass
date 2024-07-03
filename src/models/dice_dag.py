# pylint: disable=invalid-name, no-member, missing-function-docstring, too-many-branches, too-few-public-methods, unused-argument
""" DICE model from https://github.com/UsmanMahmood27/DICE """
from random import uniform, randint

import torch
from torch import nn
from torch import optim

from omegaconf import OmegaConf, DictConfig


def get_model(cfg: DictConfig, model_cfg: DictConfig):
    return DICE(model_cfg)


def get_criterion(cfg: DictConfig, model_cfg: DictConfig):
    return DICEregCEloss(model_cfg)


class DICEregCEloss:
    """Cross-entropy loss with model regularization"""

    def __init__(self, model_cfg, lambda1 = 0.01):
        self.ce_loss = nn.CrossEntropyLoss()

        self.reg_param = model_cfg.reg_param

        self.lambda1 = lambda1 # Sparsity loss weight

    def __call__(self, logits, target, model, device, DNC, DNCs):
        ce_loss = self.ce_loss(logits, target)

        # DAG-ness loss - a differentiabe measure of how distant the given DNC is from DAG space
        E = torch.linalg.matrix_exp(DNC * DNC) # (Zheng et al. 2018)
        dag_loss = torch.mean(torch.vmap(torch.trace)(E)) - DNC.shape[1]

        # Sparsity loss on DNC
        sparse_loss = self.lambda1 * torch.mean(torch.sum(torch.abs(DNC), dim=(1,2)))
        # sparse_loss = 0 

        reg_loss = torch.zeros(1).to(device)

        for name, param in model.gta_embed.named_parameters():
            if "bias" not in name:
                reg_loss += self.reg_param * torch.norm(param, p=1)

        for name, param in model.gta_attend.named_parameters():
            if "bias" not in name:
                reg_loss += self.reg_param * torch.norm(param, p=1)

        loss = ce_loss + dag_loss + sparse_loss + reg_loss
        return loss


def get_scheduler(cfg: DictConfig, model_cfg: DictConfig, optimizer):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=model_cfg.scheduler.patience,
        factor=model_cfg.scheduler.factor,
        cooldown=0,
    )
    return scheduler


def default_HPs(cfg: DictConfig):
    model_cfg = {
        "lstm": {
            "bidirectional": True,
            "num_layers": 1,
            "hidden_size": 50,
        },
        "clf": {
            "hidden_size": 64,
            "num_layers": 0,
        },
        "MHAtt": {
            "n_heads": 1,
            "head_hidden_size": 48,
            "dropout": 0.0,
        },
        "scheduler": {
            "patience": 4,
            "factor": 0.5,
        },
        "reg_param": 1e-6,
        "lr": 2e-4,
        "input_size": cfg.dataset.data_info.main.data_shape[2],
        "output_size": cfg.dataset.data_info.main.n_classes,
    }
    return OmegaConf.create(model_cfg)


def random_HPs(cfg: DictConfig):
    model_cfg = {
        "lstm": {
            "bidirectional": bool(randint(0, 1)),
            "num_layers": randint(1, 3),
            "hidden_size": randint(20, 60),
        },
        "clf": {
            "hidden_size": randint(16, 128),
            "num_layers": randint(0, 3),
        },
        "MHAtt": {
            "n_heads": randint(1, 4),
            "head_hidden_size": randint(16, 64),
            "dropout": uniform(0.0, 0.9),
        },
        "scheduler": {
            "patience": randint(1, cfg.mode.patience // 2),
            "factor": uniform(0.1, 0.8),
        },
        "reg_param": 10 ** uniform(-8, -4),
        "lr": 10 ** uniform(-5, -3),
        "input_size": cfg.dataset.data_info.main.data_shape[2],
        "output_size": cfg.dataset.data_info.main.n_classes,
    }
    return OmegaConf.create(model_cfg)


class DICE(nn.Module):
    """
    DICE model for fMRI data.
    Expected input shape: [batch_size, time_length, input_feature_size].
    Output: [batch_size, n_classes]
    """

    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        input_size = model_cfg.input_size
        output_size = model_cfg.output_size

        lstm_hidden_size = model_cfg.lstm.hidden_size
        lstm_num_layers = model_cfg.lstm.num_layers
        bidirectional = model_cfg.lstm.bidirectional

        self.lstm_output_size = (
            lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        )

        clf_hidden_size = model_cfg.clf.hidden_size
        clf_num_layers = model_cfg.clf.num_layers

        MHAtt_n_heads = model_cfg.MHAtt.n_heads
        MHAtt_hidden_size = MHAtt_n_heads * model_cfg.MHAtt.head_hidden_size
        MHAtt_dropout = model_cfg.MHAtt.dropout

        # LSTM - first block
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Classifier - last block
        clf = [
            nn.Linear(input_size**2, clf_hidden_size),
            nn.ReLU(),
        ]
        for _ in range(clf_num_layers):
            clf.append(nn.Linear(clf_hidden_size, clf_hidden_size))
            clf.append(nn.ReLU())
        clf.append(
            nn.Linear(clf_hidden_size, output_size),
        )
        self.clf = nn.Sequential(*clf)

        # Multihead attention - second block
        self.key_layer = nn.Sequential(
            nn.Linear(
                self.lstm_output_size,
                MHAtt_hidden_size,
            ),
        )
        self.value_layer = nn.Sequential(
            nn.Linear(
                self.lstm_output_size,
                MHAtt_hidden_size,
            ),
        )
        self.query_layer = nn.Sequential(
            nn.Linear(
                self.lstm_output_size,
                MHAtt_hidden_size,
            ),
        )
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=MHAtt_hidden_size,
            num_heads=MHAtt_n_heads,
            dropout=MHAtt_dropout,
            batch_first=True,
        )

        # Global Temporal Attention - third block
        self.upscale = 0.05
        self.upscale2 = 0.5

        self.HW = torch.nn.Hardswish()
        self.gta_embed = nn.Sequential(
            nn.Linear(
                input_size**2,
                round(self.upscale * input_size**2),
            ),
        )
        self.gta_norm = nn.Sequential(
            nn.BatchNorm1d(round(self.upscale * input_size**2)),
            nn.ReLU(),
        )
        self.gta_attend = nn.Sequential(
            nn.Linear(
                round(self.upscale * input_size**2),
                round(self.upscale2 * input_size**2),
            ),
            nn.ReLU(),
            nn.Linear(round(self.upscale2 * input_size**2), 1),
        )

        self.init_weight()

    def init_weight(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_in")
        for name, param in self.clf.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_in")
        for name, param in self.query_layer.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_in")
        for name, param in self.key_layer.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_in")
        for name, param in self.value_layer.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_in")
        for name, param in self.multihead_attn.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_in")
        for name, param in self.gta_embed.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.kaiming_normal_(param, mode="fan_in")
        for name, param in self.gta_attend.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.kaiming_normal_(param, mode="fan_in")

    def gta_attention(self, x, node_axis=1):
        # x.shape: [batch_size; time_length; input_feature_size * input_feature_size]
        x_readout = x.mean(node_axis, keepdim=True)
        x_readout = x * x_readout

        a = x_readout.shape[0]
        b = x_readout.shape[1]
        x_readout = x_readout.reshape(-1, x_readout.shape[2])
        x_embed = self.gta_norm(self.gta_embed(x_readout))
        x_graphattention = (self.gta_attend(x_embed).squeeze()).reshape(a, b)
        x_graphattention = self.HW(x_graphattention.reshape(a, b))
        return (x * (x_graphattention.unsqueeze(-1))).mean(node_axis)

    def multi_head_attention(self, x):
        # x.shape: [time_length * batch_size; input_feature_size; lstm_hidden_size]
        key = self.key_layer(x)
        value = self.value_layer(x)
        query = self.query_layer(x)

        attn_output, attn_output_weights = self.multihead_attn(key, value, query)

        return attn_output, attn_output_weights

    def forward(self, x):
        B, T, C = x.shape  # [batch_size, time_length, input_feature_size]

        # 1. pass input to LSTM; treat each channel as an independent single-feature time series
        x = x.permute(0, 2, 1)  # x.shape: [batch_size; input_feature_size; time_length]
        x = x.reshape(B * C, T, 1)  # x.shape: [batch_size * n_channels; time_length; 1]
        ##########################
        lstm_output, _ = self.lstm(x)
        # lstm_output.shape: [batch_size * input_feature_size; time_length; lstm_hidden_size]
        ##########################
        lstm_output = lstm_output.reshape(B, C, T, self.lstm_output_size)
        # lstm_output.shape: [batch_size; input_feature_size; time_length; lstm_hidden_size]

        # 2. pass lstm_output at each time point to multihead attention to reveal spatial connctions
        lstm_output = lstm_output.permute(2, 0, 1, 3)
        # lstm_output.shape: [time_length; batch_size; input_feature_size; lstm_hidden_size]
        lstm_output = lstm_output.reshape(T * B, C, self.lstm_output_size)
        # lstm_output.shape: [time_length * batch_size; input_feature_size; lstm_hidden_size]
        ##########################
        _, attn_weights = self.multi_head_attention(lstm_output)
        # attn_weights.shape: [time_length * batch_size; input_feature_size; input_feature_size]
        ##########################
        attn_weights = attn_weights.reshape(T, B, C, C)
        # attn_weights.shape: [time_length; batch_size; input_feature_size; input_feature_size]
        attn_weights = attn_weights.permute(1, 0, 2, 3)
        # attn_weights.shape: [batch_size; time_length; input_feature_size; input_feature_size]

        weights = attn_weights

        # 3. pass attention weights to a global temporal attention to obrain global graph
        attn_weights = attn_weights.reshape(B, T, -1)
        # attn_weights.shape: [batch_size; time_length; input_feature_size * input_feature_size]
        ##########################
        FC = self.gta_attention(attn_weights)
        # FC.shape: [batch_size; input_feature_size * input_feature_size]
        ##########################

        # 4. Pass learned graph to the classifier to get predictions
        logits = self.clf(FC)
        # logits.shape: [batch_size; n_classes]

        return logits, FC.reshape(B, C, C), weights
