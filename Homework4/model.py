import torch
import torch.nn as nn
import numpy as np


class SDFMLP(nn.Module):
    """基础MLP模型，输入3D坐标，输出1D SDF值。"""

    def __init__(
        self,
        in_dim=3,
        hidden_dim=256,
        num_layers=8,
        activation="relu",
        skip_layers=[4],
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip_layers = skip_layers

        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "sine":
            self.act = torch.sin
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(in_dim, hidden_dim))
            elif i in skip_layers:
                self.layers.append(nn.Linear(hidden_dim + in_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, 1)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        """
        Args:
            x: (B, 3) 查询点坐标
        Returns:
            sdf: (B, 1) SDF值
        """
        h = x
        for i, layer in enumerate(self.layers):
            if i in self.skip_layers:
                h = torch.cat([h, x], dim=-1)
            h = layer(h)
            if self.act is torch.sin:
                h = self.act(h)
            else:
                h = self.act(h)
        sdf = self.output_layer(h)
        return sdf


class FourierFeatureMLP(nn.Module):
    """Fourier Feature位置编码 + MLP。"""

    def __init__(
        self,
        in_dim=3,
        hidden_dim=256,
        num_layers=8,
        activation="relu",
        skip_layers=[4],
        mapping_size=10,
        sigma=10.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.mapping_size = mapping_size
        self.sigma = sigma

        # 随机高斯矩阵 B: (in_dim, mapping_size)
        B = torch.randn(in_dim, mapping_size) * sigma
        self.register_buffer("B", B)

        self.mlp = SDFMLP(
            in_dim=2 * mapping_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            skip_layers=skip_layers,
        )

    def fourier_features(self, x):
        """
        Args:
            x: (B, in_dim)
        Returns:
            feat: (B, 2 * mapping_size)
        """
        # x @ B: (B, mapping_size)
        proj = 2 * np.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, x):
        feat = self.fourier_features(x)
        return self.mlp(feat)
