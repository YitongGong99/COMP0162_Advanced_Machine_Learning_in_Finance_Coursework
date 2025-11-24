# -*- coding: utf-8 -*-
import torch
from torch import nn, optim

# Training Data Generation
# ---------------------------------------------------------------------------------------------------------------------

def random_routes(
        S0_: torch.Tensor,
        batch_size: int,
        time_to_maturity: float | int,
        num_interval: int,
        risk_free_rate: float,
        covariance: torch.Tensor,
        device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    generate random price routes using SDE
    :param S0_: initial prices
    :param batch_size: number of samples
    :param time_to_maturity: time to maturity
    :param num_interval: number of small intervals
    :param risk_free_rate: risk-free rate
    :param covariance: covariance matrix of assets
    :param device: device to run
    :return: samples routes with size (batch_size, num_interval + 1, asset_dim)
    """
    if S0_.dim() > 1:
        raise ValueError("S0 can only be a tensor with shape (N)")
    asset_dim = S0_.shape[0]
    dt = time_to_maturity / num_interval
    std = torch.linalg.cholesky(covariance)

    dW_t = torch.distributions.Normal(0, 1).sample(
        (batch_size, num_interval, asset_dim)
    ).to(device) * dt ** 0.5

    S_ = torch.zeros(batch_size, num_interval + 1, asset_dim, device=device)
    S_[:, 0, :] = S0_
    for idx in range(num_interval):
        S_[:, idx + 1, :] = S_[:, idx, :] * torch.exp(risk_free_rate * dt + dW_t[:, idx, :] @ std)
    return dW_t, S_


def payoff(price: torch.Tensor, strike: float, kind: str) -> torch.Tensor:
    """

    :param price: sample routes
    :param strike: strike price
    :param kind: call or put
    :return: payoff of each sample
    """
    if kind not in ["call", "put"]:
        raise ValueError("kind can only be 'call' or 'put'")
    direc = 1 if kind == "call" else -1
    return torch.nn.functional.relu((price[:, -1, :].mean(-1) - strike) * direc)


def monte_carlo_pricing(price_, risk_free_rate, time_to_maturity, payoff_, strike: float, kind: str) -> torch.Tensor:
    payoff_maturity = payoff_(price_, strike, kind)
    return payoff_maturity.mean() * torch.exp(torch.tensor(- risk_free_rate * time_to_maturity))


# PINN Deep BSDE Solver Architectures
# --------------------------------------------------------------------------------------------------------------------


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim=10, device='cuda'):
        super().__init__()
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.to(self.device)

    def forward(self, t):  # t: [B, N, 1]
        return self.fc(t)


class YNet(nn.Module):
    def __init__(self, asset_dim, latent_dim, device="cuda"):
        super(YNet, self).__init__()
        self.device = device
        self.asset_dim = asset_dim
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Softplus(),
            nn.Linear(latent_dim, 1)
        )
        self.to(self.device)

    def forward(self, x):
        return self.net(x)


class ZNet(nn.Module):
    def __init__(self, asset_dim, latent_dim, device="cuda"):
        super(ZNet, self).__init__()
        self.device = device
        self.asset_dim = asset_dim
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Softplus(),
            nn.Linear(latent_dim, asset_dim)
        )
        self.to(self.device)

    def forward(self, x):
        return self.net(x)


class BSDE_FCNN(nn.Module):
    def __init__(self, asset_dim, hidden_dim, latent_dim, time_embedding_dim, device="cuda"):
        super(BSDE_FCNN, self).__init__()
        self.device = device
        self.asset_dim = asset_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(asset_dim + time_embedding_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.t_embed = TimeEmbedding(embed_dim=time_embedding_dim,device=self.device)
        self.to(self.device)

    def forward(self, T_: torch.Tensor, S_: torch.Tensor) -> torch.Tensor:
        """

        :param T_: shape (num_interval + 1, )
        :param S_: shape (batch_size, num_interval + 1, asset_dim)
        :return: latent: shape (batch_size, latent_dim)
        """
        if T_.ndim != 3:
            raise ValueError("T_ must be 3-D")
        if S_.ndim != 3:
            raise ValueError("S_ must be 3-D (batch_size, num_interval + 1, asset_dim)")
        Input = torch.cat((self.t_embed(T_), S_), dim=-1)
        return self.net(Input)

    @property
    def num_params(self):
        return sum([x.numel() for x in self.parameters()])


class BSDE_GRU(nn.Module):
    def __init__(self, asset_dim, hidden_dim, latent_dim, time_embedding_dim, device: str | torch.device="cuda"):
        super(BSDE_GRU, self).__init__()
        self.device = device
        self.asset_dim = asset_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.GRU = nn.GRU(input_size=asset_dim + time_embedding_dim, hidden_size=hidden_dim, num_layers=1).to(self.device)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=latent_dim).to(self.device)
        self.t_embed = TimeEmbedding(embed_dim=time_embedding_dim,device=self.device)
        self.to(self.device)

    def forward(self, T_: torch.Tensor, S_: torch.Tensor) -> torch.Tensor:
        """

        :param T_: shape (num_interval + 1, )
        :param S_: shape (batch_size, num_interval + 1, asset_dim)
        :return: latent: shape (batch_size, latent_dim)
        """
        if T_.ndim != 3:
            raise ValueError("T_ must be 3-D")
        if S_.ndim != 3:
            raise ValueError("S_ must be 3-D (batch_size, num_interval + 1, asset_dim)")
        Input = torch.cat((self.t_embed(T_), S_), dim=-1)
        return self.fc(self.GRU(Input)[0])

    @property
    def num_params(self):
        return sum([x.numel() for x in self.parameters()])


class BSDE_RNN(nn.Module):
    def __init__(self, asset_dim, hidden_dim, latent_dim, time_embedding_dim, device: str | torch.device="cuda"):
        super(BSDE_RNN, self).__init__()
        self.device = device
        self.asset_dim = asset_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.RNN = nn.RNN(input_size=asset_dim + time_embedding_dim, hidden_size=hidden_dim, num_layers=2).to(self.device)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=latent_dim).to(self.device)
        self.t_embed = TimeEmbedding(embed_dim=time_embedding_dim,device=self.device)
        self.to(self.device)

    def forward(self, T_: torch.Tensor, S_: torch.Tensor) -> torch.Tensor:
        """

        :param T_: shape (num_interval + 1, )
        :param S_: shape (batch_size, num_interval + 1, asset_dim)
        :return: latent: shape (batch_size, latent_dim)
        """
        if T_.ndim != 3:
            raise ValueError("T_ must be 3-D")
        if S_.ndim != 3:
            raise ValueError("S_ must be 3-D (batch_size, num_interval + 1, asset_dim)")
        Input = torch.cat((self.t_embed(T_), S_), dim=-1)
        return self.fc(self.RNN(Input)[0])

    @property
    def num_params(self):
        return sum([x.numel() for x in self.parameters()])


class BSDE_LSTM(nn.Module):
    def __init__(self, asset_dim, hidden_dim, latent_dim, time_embedding_dim, device: str | torch.device="cuda"):
        super(BSDE_LSTM, self).__init__()
        self.device = device
        self.asset_dim = asset_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.LSTM = nn.LSTM(input_size=asset_dim + time_embedding_dim, hidden_size=hidden_dim, num_layers=1).to(self.device)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=latent_dim).to(self.device)
        self.t_embed = TimeEmbedding(embed_dim=time_embedding_dim,device=self.device)
        self.to(self.device)

    def forward(self, T_: torch.Tensor, S_: torch.Tensor) -> torch.Tensor:
        """

        :param T_: shape (num_interval + 1, )
        :param S_: shape (batch_size, num_interval + 1, asset_dim)
        :return: latent: shape (batch_size, latent_dim)
        """
        if T_.ndim != 3:
            raise ValueError("T_ must be 3-D")
        if S_.ndim != 3:
            raise ValueError("S_ must be 3-D (batch_size, num_interval + 1, asset_dim)")
        Input = torch.cat((self.t_embed(T_), S_), dim=-1)
        return self.fc(self.LSTM(Input)[0])

    @property
    def num_params(self):
        return sum([x.numel() for x in self.parameters()])


class BSDE_Net(nn.Module):
    def __init__(self, model, model_name, asset_dim, hidden_dim, latent_dim, time_embedding_dim, trained=False, device: str | torch.device="cuda"):
        super(BSDE_Net, self).__init__()
        self.device = device
        self.net_config = {
            "asset_dim": asset_dim,
            "latent_dim": latent_dim,
            "time_embedding_dim": time_embedding_dim,
            "hidden_dim": hidden_dim,
            "device": self.device
        }
        self.net = model(**self.net_config)
        self.YNet = YNet(asset_dim, latent_dim, device)
        self.ZNet = ZNet(asset_dim, latent_dim, device)
        self.trained = trained
        self.model_name = model_name
        if trained:
            self.load_state_dict(torch.load(f"./params/Best_{model_name}.pth"))

    def forward(self, T_: torch.Tensor, S_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent_result = self.net(T_, S_)
        return self.YNet(latent_result), self.ZNet(latent_result)


# Loss Functions
# ---------------------------------------------------------------------------------------------------------------------


def dynamic_residue(Y_: torch.Tensor, Z_: torch.Tensor, dW_t: torch.Tensor, time_grid: torch.Tensor, risk_free_rate: float):
    assert Y_.ndim == Z_.ndim == 3
    dt = (time_grid.max() - time_grid.min()) / (len(time_grid) - 1)
    Y_ = Y_.squeeze(-1)
    Loss = Y_[:, :-1] - Y_[:, 1:] - risk_free_rate * Y_[:, :-1] * dt + (Z_[:, :-1, :] * dW_t).sum(-1)
    Loss = torch.pow(Loss, 2).mean(-1).sum()
    return Loss


def terminal_residue(Y_, price: torch.Tensor, payoff_, strike, kind):
    True_boundary = payoff_(price=price, strike=strike, kind=kind)
    Y_T = Y_.squeeze()[:, -1]
    return nn.MSELoss(reduction="sum")(True_boundary, Y_T)


def pinn_residue(model, price: torch.Tensor, time_grid: torch.Tensor, risk_free_rate: float, cov_matrix: torch.Tensor):
    batch_size, num_interval, asset_dim = price.shape
    price_ = price.detach().clone().requires_grad_(True)
    time_grid_ = time_grid.detach().clone().requires_grad_(True)
    Y_, _ = model(time_grid_, price_)
    dY_dt = torch.autograd.grad(Y_, time_grid_, create_graph=True, grad_outputs=torch.ones_like(Y_))[0]
    dY_dS = torch.autograd.grad(Y_, price_, create_graph=True, grad_outputs=torch.ones_like(Y_))[0]
    d2Y_dS2 = torch.concat(
        [
            torch.autograd.grad(
                dY_dS[:, :, i].unsqueeze(-1),
                price_,
                create_graph=True,
                grad_outputs=torch.ones_like(dY_dt)
            )[0].unsqueeze(-1)
            for i in range(asset_dim)
        ], dim=-1
    )
    d2Y_dS2 = 0.5 * (d2Y_dS2 + d2Y_dS2.transpose(-2, -1))
    diag_price = torch.diag_embed(price)
    _1d = torch.matmul(price.unsqueeze(-1).transpose(-2, -1), dY_dS.unsqueeze(-1)).squeeze(-1) * risk_free_rate
    _2d = 1/2 * (diag_price * cov_matrix * diag_price * d2Y_dS2).diagonal(dim1=-2, dim2=-1).sum(-1).unsqueeze(-1)
    loss_matrix = dY_dt + _1d + _2d - risk_free_rate * Y_
    return (loss_matrix.unsqueeze(-1) ** 2).mean(-1).sum()


# Model Training (Adam)
# ---------------------------------------------------------------------------------------------------------------------


def training_bsde_solver(model, model_name ,epochs, dwt, time_grid, risk_free_rate, price, cov_matrix, payoff_, strike, kind, pinn_on=True, **kwargs):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=kwargs.get("lr", 0.001))
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=kwargs.get("step_size", int(epochs / 10)), gamma=kwargs.get("gamma", 0.1))
    Loss = []
    Prices = []
    best_loss = float("inf")
    prev_loss = float("inf")

    for epoch in range(epochs):
        optimizer.zero_grad()
        Y_, Z_ = model(time_grid, price)
        loss_terminal = terminal_residue(Y_, price, payoff_, strike, kind) * (0.1 if pinn_on else 1)
        loss_dynamic = dynamic_residue(Y_, Z_, dwt, time_grid[0], risk_free_rate) * 1
        loss = loss_dynamic + loss_terminal
        if pinn_on:
            with torch.backends.cudnn.flags(enabled=False):
                loss_pinn = pinn_residue(model, price, time_grid, risk_free_rate, cov_matrix) * 10
                loss += loss_pinn
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if loss.item() <= best_loss:
            best_loss = loss.item()

        Y_0, _ = model.forward(T_=time_grid[:1, :1, ], S_=price[:1, :1, ])
        print(
                f"Epoch : {epoch + 1}/{epochs} | "
                f"Loss : {loss: .6f} | "
                f"Dynamic Loss : {loss_dynamic:.4f} | "
                f"Terminal Loss : {loss_terminal:.4f} | " +
                (f"Pinn Loss : {loss_pinn:.4f} | " if pinn_on else "") +
                f"Y(0, 100): {Y_0[0, 0, 0]: .4f} | "
        )
        Loss.append(loss.item())
        Prices.append(Y_0[0,0,0].item())

        if abs(Loss[-1] - prev_loss) < kwargs.get("min_change", 1e-6):
            break
        prev_loss = loss.item()
    if kwargs.get("save_params", True):
        torch.save(model.state_dict(), f"./params/Best_{model_name}.pth")
    return Loss, Prices


if __name__ == '__main__':
    DEVICE = torch.device('cuda')
    EPOCHS = 10000
    ASSET_DIM = 10
    LATENT_DIM = 10
    TIME_TO_MATURITY = 1 / 12
    NUM_INTERVAL = 10
    TIME_EMBEDDING_DIM = 1
    DT = TIME_TO_MATURITY / NUM_INTERVAL
    BATCH_SIZE = 10 ** 4
    RISK_FREE_RATE = 0.02
    STRIKE = 100
    KIND = "call"
    COV = 0.01 * torch.eye(ASSET_DIM).to(DEVICE)
    S0 = torch.ones(ASSET_DIM, device=DEVICE) * 100

    T = torch.linspace(0, 1, NUM_INTERVAL + 1, device=DEVICE).repeat(BATCH_SIZE, 1).unsqueeze(-1)

    torch.manual_seed(42)
    dW, S = random_routes(
        S0_=S0,
        batch_size=BATCH_SIZE,
        time_to_maturity=TIME_TO_MATURITY,
        num_interval=NUM_INTERVAL,
        risk_free_rate=RISK_FREE_RATE,
        covariance=COV,
        device=DEVICE
    )

    FCNN_net = BSDE_Net( model=BSDE_FCNN, model_name="BSDE_FCNN_1d", asset_dim=ASSET_DIM, hidden_dim=35, latent_dim=LATENT_DIM, time_embedding_dim=TIME_EMBEDDING_DIM, trained=False, device=DEVICE,)
    RNN_net = BSDE_Net( model=BSDE_RNN, model_name="BSDE_RNN_1d", asset_dim=ASSET_DIM, hidden_dim=30, latent_dim=LATENT_DIM, time_embedding_dim=TIME_EMBEDDING_DIM, trained=False, device=DEVICE,)
    GRU_net = BSDE_Net( model=BSDE_GRU, model_name="BSDE_GRU_1d", asset_dim=ASSET_DIM, hidden_dim=25, latent_dim=LATENT_DIM, time_embedding_dim=TIME_EMBEDDING_DIM, trained=False, device=DEVICE,)
    LSTM_net = BSDE_Net( model=BSDE_LSTM, model_name="BSDE_LSTM_1d", asset_dim=ASSET_DIM, hidden_dim=22, latent_dim=LATENT_DIM, time_embedding_dim=TIME_EMBEDDING_DIM, trained=False, device=DEVICE,)

    net = FCNN_net
    if not net.trained:
        losses = training_bsde_solver(
            model=net,
            model_name=net.model_name,
            epochs=EPOCHS,
            dwt=dW,
            time_grid=T,
            risk_free_rate=RISK_FREE_RATE,
            cov_matrix=COV,
            price=S,
            payoff_=payoff,
            strike=STRIKE,
            kind=KIND,
            pinn_on=False,
            lr=1e-3,
            step_size=100,
            gamma=0.9,
        )

    Y, Z = net(T, S)
    print(S[0],Y[0])
