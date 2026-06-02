import torch
import torch.nn as nn
import numpy as np


class SDFMLP(nn.Module):
    """基础MLP模型，输入3D坐标，输出1D SDF值。

    参考IGR [Gropp et al., ICML 2020] 实现 geometric initialization
    和 skip connection 缩放，提升训练稳定性和收敛速度。
    """

    def __init__(
        self,
        in_dim=3,
        hidden_dim=512,
        num_layers=10,
        activation="relu",
        skip_layers=[5],
        geometric_init=True,
        radius_init=1.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip_layers = skip_layers
        self.geometric_init = geometric_init
        self.radius_init = radius_init

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
        if geometric_init:
            self._geometric_init_weights()
        else:
            self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def _geometric_init_weights(self):
        """IGR 风格几何初始化：让网络初始近似一个球面 SDF。"""
        # 隐藏层使用 He 初始化
        for layer in self.layers:
            nn.init.constant_(layer.bias, 0.0)
            out_dim = layer.weight.shape[0]
            nn.init.normal_(layer.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(out_dim))

        # 输出层：让初始输出接近 sphere SDF = |x| - radius_init
        # 这样初始零等值面为半径 radius_init 的球
        nn.init.normal_(
            self.output_layer.weight,
            mean=np.sqrt(np.pi) / np.sqrt(self.output_layer.weight.shape[1]),
            std=1e-5,
        )
        nn.init.constant_(self.output_layer.bias, -self.radius_init)

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
                # 参考IGR：skip connection 除以 sqrt(2) 保持方差稳定
                h = torch.cat([h, x], dim=-1) / np.sqrt(2)
            h = layer(h)
            if self.act is torch.sin:
                h = self.act(h)
            else:
                h = self.act(h)
        sdf = self.output_layer(h)
        return sdf


class FourierFeatureMLP(nn.Module):
    """Fourier Feature位置编码 + MLP。

    参考 "Fourier Features Let Networks Learn High Frequency Functions
    in Low Dimensional Domains" (Tancik et al., NeurIPS 2020)
    """

    def __init__(
        self,
        in_dim=3,
        hidden_dim=512,
        num_layers=10,
        activation="relu",
        skip_layers=[5],
        mapping_size=64,
        sigma=5.0,
        geometric_init=True,
        radius_init=1.0,
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
            geometric_init=geometric_init,
            radius_init=radius_init,
        )

    def fourier_features(self, x):
        """
        Args:
            x: (B, in_dim)
        Returns:
            feat: (B, 2 * mapping_size)
        """
        proj = 2 * np.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, x):
        feat = self.fourier_features(x)
        return self.mlp(feat)
