import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils
import cvxopt


class MetaEgOnsAgent(Agent):
    def __init__(self, config, data_loader, trade_log=None):
        super().__init__(config, data_loader)
        self.exchange = config['exchange']
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader, 
                                timetable, config, 2 * config['window'] + 1, True)
        self.instruments_list = config["instruments_list"]
        self.n_steps = 0
        self.exchange = config['exchange']
        self.n_inst = len(self.instruments_list)
        self.learning_rate = config['learning_rate']
        # ons parameters
        self.eta = config['mixture']
        self.beta = config['tradeoff']
        self.delta = config['heuristic_tuning']
        self.A = np.eye(self.n_inst)
        self.b = np.zeros((self.n_inst, 1))
        # anticor parameters
        self.window = config['window']
        self.double_window = 2 * self.window + 1
        # meta parameters
        self.update_method = config["update_method"]
        if self.update_method not in {"eg", "ons"}:
            print("Update method is not available.")
            print("Available methods: eg, ons.")
            exit(1)
        self.meta_update_parameter = config["meta_update_parameter"]
        self.meta_num = 3
        self.meta_weights = np.ones(self.meta_num) / self.meta_num
        self.predictions = None
        if config.get("meta_epsilon") is not None:
            self.meta_epsilon = config["meta_epsilon"]
        else:
            self.meta_epsilon = 0
        self.A_meta = np.eye(self.meta_num) * self.meta_epsilon
        self.last_portfolio = np.ones(self.n_inst) / self.n_inst

    def eg_next_weight(self, last_x):
        weight = self.last_portfolio * np.exp(self.learning_rate * \
                        last_x / (last_x @ self.last_portfolio))
        return weight / weight.sum()

    def find_projection_to_simplex(self, x, M):
        n = M.shape[0]
        P = cvxopt.matrix(2 * M)
        q = cvxopt.matrix(-2 * M @ x)
        G = cvxopt.matrix(-np.eye(n))
        h = cvxopt.matrix(np.zeros((n, 1)))
        A = cvxopt.matrix(np.ones((1, n)))
        b = cvxopt.matrix(1.)
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        return np.squeeze(sol['x'])

    def ons_next_weight(self, x):
        N = len(x)
        grad = x / (x @ self.last_portfolio)
        grad = grad.reshape(-1, 1)
        hessian = -1 * (grad @ grad.T)
        self.A += (-1) * hessian
        self.b += (1 + 1 / self.beta) * grad
        A_inv = np.linalg.inv(self.A)
        B = self.delta * A_inv @ self.b
        weight = self.find_projection_to_simplex(B, self.A)
        weight = (1 - self.eta) * weight + self.eta * np.ones(N) / N
        return weight / sum(weight)

    def anticor_expert(self, data, weight_o, w):
        T, N = data.shape
        weight = weight_o.copy()

        if T < 2 * w:
            return weight

        LX_1 = np.log(data[T - 2 * w: T - w])
        LX_2 = np.log(data[T - w: T])

        mu_1 = np.mean(LX_1, axis=0)
        mu_2 = np.mean(LX_2, axis=0)
        M_cov = np.zeros((N, N))
        M_cor = np.zeros((N, N))
        n_LX_1 = LX_1 - np.repeat(mu_1.reshape(1, -1), w, axis=0)
        n_LX_2 = LX_2 - np.repeat(mu_2.reshape(1, -1), w, axis=0)

        Std_1 = np.diag(n_LX_1.T @ n_LX_1) / (w - 1)
        Std_2 = np.diag(n_LX_2.T @ n_LX_2) / (w - 1)
        Std_12 = Std_1.reshape(-1, 1) @ Std_2.reshape(1, -1)
        M_cov = n_LX_1.T @ n_LX_2 / (w - 1)

        M_cor[Std_12 == 0] = 0
        M_cor[Std_12 != 0] = M_cov[Std_12 != 0] / np.sqrt(Std_12[Std_12 != 0])

        claim = np.zeros((N, N))
        w_mu_2 = np.repeat(mu_2.reshape(-1, 1), N, axis=1)
        w_mu_1 = np.repeat(mu_2.reshape(1, -1), N, axis=0)

        s_12 = (w_mu_2 >= w_mu_1) & (M_cor > 0)
        claim[s_12] += M_cor[s_12]

        diag_M_cor = np.diag(M_cor)

        cor_1 = np.repeat(-diag_M_cor.reshape(-1, 1), N, axis=1)
        cor_2 = np.repeat(-diag_M_cor.reshape(1, -1), N, axis=0)
        cor_1 = np.maximum(0, cor_1)
        cor_2 = np.maximum(0, cor_2)
        claim[s_12] += cor_1[s_12] + cor_2[s_12]

        transfer = np.zeros((N, N))
        sum_claim = np.repeat(np.sum(claim, axis=1).reshape(-1, 1), N, axis=1)
        s_1 = np.abs(sum_claim) > 0
        w_weight_o = np.repeat(weight_o.reshape(-1, 1), N, axis=1)
        transfer[s_1] = w_weight_o[s_1] * claim[s_1] / sum_claim[s_1]

        transfer_ij = transfer.T - transfer
        weight -= np.sum(transfer_ij, axis=0)
        return weight

    def anticor_kernel(self, data):
        weight = self.anticor_expert(data, self.last_portfolio, self.window)
        return weight / weight.sum()

    def update_distribution_gradient(self, true_price_relatives):
        f_grad = -self.predictions.T @ true_price_relatives
        denom = true_price_relatives @ self.predictions @ self.meta_weights
        f_grad /= denom
        print("f_grad:", f_grad)
        print("denom:", denom)
        u_bar = true_price_relatives.max() / true_price_relatives.min()
        loss = 0.5 * (f_grad / u_bar + np.ones(self.meta_num))
        print("loss:", loss)
        self.meta_weights *= np.exp(-self.meta_update_parameter * loss)
        self.meta_weights /= self.meta_weights.sum()
        print("weights:", self.meta_weights)

    def update_distribution_newton(self, true_price_relatives):
        u_bar = true_price_relatives.max() / true_price_relatives.min()
        alpha = 1.
        beta = 1. / 8 / u_bar
        f_grad = -self.predictions.T @ true_price_relatives
        denom = true_price_relatives @ self.predictions @ self.meta_weights
        f_grad /= denom
        self.A_meta += f_grad @ f_grad.T
        A_inv = np.linalg.inv(self.A_meta)
        self.meta_weights -= 2 / beta * A_inv @ f_grad
        self.meta_weights = self.find_projection_to_simplex(self.meta_weights, self.A_meta)
        self.meta_weights /= self.meta_weights.sum()
        print("weights:", self.meta_weights)

    def online_update(self, true_price_relatives):
        if self.update_method == "eg":
            self.update_distribution_gradient(true_price_relatives)
        elif self.update_method == "ons":
            self.update_distribution_newton(true_price_relatives)
        else:
            print("Wrong update method.")
            exit(1)

    def predict_next_weights(self, data_price_relatives):
        if self.n_steps > 1:
            self.online_update(data_price_relatives[-1])
        weights_eg = self.eg_next_weight(data_price_relatives[-1])
        weights_ons = self.ons_next_weight(data_price_relatives[-1])
        weights_anticor = self.anticor_kernel(data_price_relatives)
        self.predictions = np.array([
            weights_eg,
            weights_ons,
            weights_anticor
        ]).T
        print("\npredictions:\n", self.predictions, '\n')
        weights_prediction = self.predictions @ self.meta_weights
        return weights_prediction

    def compute_portfolio(self, epoch):
        self.n_steps += 1
        data_price_relatives = self.data_extractor(epoch)
        day_weight = self.predict_next_weights(data_price_relatives)
        print("meta_eg_ons weights:", day_weight)
        self.last_portfolio = day_weight
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = day_weight[i]
        return preds_dict
