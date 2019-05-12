import numpy as np
import pandas as pd
import scipy.optimize as optimize
from scipy.spatial.distance import cdist, euclidean
from .calendar_moex import CalendarMOEX


def compute_L1_median(X, maxiter=200, tol=1e-5):
    """
    Finds L1 median to historical prices
    
    :param X: prices
    :param maxiter: max number of iterations
    :param tol: toleration level
    Returns predicted prices.
    """
    y = np.median(X, 0)
    for i in range(maxiter):
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]
        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)
        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y
        if euclidean(y, y1) < tol:
            return y1
        y = y1
    return y


def optimize_weights(X, **kwargs):
    """
    Finds best constant rebalanced portfolio weights.
    :param X: Prices in ratios
    :params kwargs: additional parameters to scipy optimizer.
    """

    x_0 = np.ones(X.shape[1]) / float(X.shape[1])
    objective = lambda b: -np.sum(np.log(np.maximum(np.dot(X - 1, b) + 1, 0.0001)))
    cons = ({'type': 'eq', 'fun': lambda b: 1 - sum(b)},)
    while True:
        res = optimize.minimize(objective, x_0, bounds=[(0., 1.)] * len(x_0),
                                constraints=cons, method='slsqp', **kwargs)
        # result can be out-of-bounds -> try it again
        EPS = 1e-7
        if (res.x < 0. - EPS).any() or (res.x > 1. + EPS).any():
            X = X + np.random.randn(1)[0] * 1e-5
            print('Optimal weights not found, trying again...')
            continue
        elif res.success:
            break
        else:
            if np.isnan(res.x).any():
                print('Solution does not exist, use zero weights.')
                res.x = np.zeros(X.shape[1])
            else:
                print('Converged, but not successfully.')
            break
    return res.x


def simplex_projection(v, b=1):
    """
    Simplex_Projection Projects point onto simplex of specified radius.
    w = simplex_projection(v, b) returns the vector w which is the solution
    to the following constrained minimization problem:
        
        min   ||w - v||_2
        s.t.  sum(w) <= b, w >= 0.

    That is, performs Euclidean projection of v to the positive simplex of
    radius b.
    """
    if np.all(v == v[0]):
        return np.ones(v.shape[0]) / v.shape[0]
    if b < 0:
        print('Radius of simplex is negative: %2.3f\n', b)
        return None
    v = np.minimum(v, 1e15)
    v = (v > 0) * v
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)
    z = (u > (sv - b) / range(1, len(u) + 1))
    non_neg_indices = np.argwhere(z != 0)
    if len(non_neg_indices):
        rho = non_neg_indices[-1, -1]
    else:
        rho = 0
    # theta = np.maximum(0, (sv[rho] - b) / (rho + 1))
    theta = (sv[rho] - b) / (rho + 1)
    res = np.maximum(v - theta, 0)
    return res


def make_ratios(data):
    """
    Converts prices to ratios.
    
    :param data: numpy.ndarray containing prices.
    """
    x0 = np.ones(data.shape[1])
    x = data[1:] / data[:-1]
    return np.vstack((x0, x))


class WeightsProjection:
    def __init__(self, config):
        if config.get("short_flag") is not None:
            self.short_flag = config["short_flag"]
        else:
            self.short_flag = False
        if config.get("neutralize_flag") is not None:
            self.neutralize_flag = config["neutralize_flag"]
        else:
            self.neutralize_flag = False
        if config.get("projection_method") is not None:
            self.projection_method = config["projection_method"]
            self.check_projection_method(self.projection_method)
        else:
            self.projection_method = "simplex"
        if config.get("top_amount") is not None:
            self.top_amount = config["top_amount"]
        else:
            self.top_amount = 1

    def __call__(self, weights):
        if np.all(weights == weights[0]):
            return np.ones(weights.shape[0]) / weights.shape[0]
        if self.projection_method == "simplex":
            if self.short_flag:
                weights = self.simplex_projection_with_short(weights)
            else:
                weights = simplex_projection(weights)
                weights /= weights.sum()
        elif self.projection_method == "uniform":
            weights = self.uniform_projection(weights)
        elif self.projection_method == "top_k":
            weights = self.top_projection(weights)
        elif self.projection_method == "identical":
            pass
        else:
            raise Exception("Wrong projection_method")
        return weights

    def check_projection_method(self, projection_method):
        if projection_method not in {"simplex", "uniform", "top_k", "identical"}:
            raise Exception("Wrong projection_method")

    def top_projection(self, weights, top_amount=1, min_value=0, max_value=1):
        indices_sorted = weights.argsort()
        indices_long = indices_sorted[-self.top_amount:]
        weights_long = weights[indices_long]
        if self.short_flag:
            indices_short = indices_sorted[:self.top_amount]
            weights_short = weights[indices_short]
        if self.neutralize_flag:
            weights_long /= weights_long.sum()
            if self.short_flag:
                weights_short /= np.abs(weights_short).sum()
        final_weights = np.zeros(weights.shape[0])
        final_weights[indices_long] = weights_long
        if self.short_flag:
            final_weights[indices_short] = weights_short
        return final_weights / np.abs(final_weights).sum()

    def uniform_projection(self, weights, min_value=0, max_value=1):
        if self.short_flag:
            min_value = -1
        min_weight = weights.min()
        weight_std = (weights - min_weight) / (weights.max() - min_weight)
        weight_scaled = weight_std * (max_value - min_value) + min_value
        return weight_scaled / np.abs(weight_scaled).sum()

    def simplex_projection_with_short(self, weights):
        weight_scaled = np.zeros(weights.shape[0])
        indices_short = weights <= weights.mean()
        indices_long = weights > weights.mean()
        weights_short = -simplex_projection(-weights[indices_short])
        weights_long = simplex_projection(weights[indices_long])
        weight_scaled[indices_short] = weights_short
        weight_scaled[indices_long] = weights_long
        return weight_scaled / np.abs(weight_scaled).sum()


class ExchangeTimetable:
    def __init__(self, exchange):
        if exchange == "moex":
            self.timetable = CalendarMOEX()
        else:
            self.timetable = None

    def get_last_epochs(self, epoch, step, amount):
        if self.timetable is None:
            return [epoch - step * i for i in range(amount)][::-1]
        output_times = []
        length = 0
        new_epoch = epoch
        while length < amount:
            if self.timetable[new_epoch]:
                output_times.append(new_epoch)
                length += 1
            new_epoch -= step
        return output_times[::-1]


class DataExtractor:
    def __init__(self, data_loader, timetable, config, amount, 
                return_price_relatives=None, relevant_columns=None, return_pandas=False):
        self.data_loader = data_loader
        self.timetable = timetable
        self.step = config["step"]
        if config.get("use_risk_free") is not None:
            self.use_risk_free = config["use_risk_free"]
        else:
            self.use_risk_free = False
        self.amount = amount
        self.return_price_relatives = return_price_relatives
        self.return_pandas = return_pandas
        if relevant_columns is not None:
            self.relevant_columns = relevant_columns
        else:
            self.relevant_columns = [
                ">".join([
                    config["exchange"],
                    label,
                    config["candles_res"],
                    config["price_label"]
                ])
                for label in config["instruments_list"]
            ]

    def __call__(self, epoch, amount=None, relatives=None):
        if amount is None:
            amount = self.amount
        output_times = self.timetable.get_last_epochs(epoch, self.step, amount)
        data_dict = self.data_loader.get_data(output_times)
        data = []
        for epoch, epoch_data in data_dict.items():
            new_epoch_data = [epoch_data[col] for col in self.relevant_columns]
            data.append(new_epoch_data)
        data = pd.DataFrame(data, columns=self.relevant_columns)
        data.fillna(method="pad", inplace=True)
        data.fillna(method="backfill", inplace=True)
        if self.return_pandas:
            return data
        data.reset_index(inplace=True, drop=True)
        data = np.array(data)
        if self.use_risk_free:
            data = np.hstack(
                [data, np.ones((data.shape[0], 1))]
            )
        if relatives is not None:
            if relatives:
                return data[1:] / data[:-1]
            else:
                return data
        else:
            if self.return_price_relatives:
                return data[1:] / data[:-1]
            else:
                return data


class Arima:
    def __init__(self, window=7, epsilon=10**-0.5, learning_rate=1.75, method="ons",
                lr_decrease_power=None, init_weights=None, verbose=False):
        self.epsilon = epsilon
        if init_weights is None:
            self.weights = np.arange(1, window + 1) / 10
        else:
            self.weights = init_weights.copy()
        self.learning_rate = learning_rate
        self.window = window
        self.method = method
        self.A_trans = np.eye(self.window) / epsilon
        self.A = np.eye(self.window) * epsilon
        self.lr_decrease_power = lr_decrease_power
        self.verbose = verbose

    def predict_next_price_arima_ogd(self, X):
        diff = self.weights @ X[:-1] - X[-1]
        grad = 2 * X[:-1] * diff
        if self.verbose:
            print("pred:\t", self.weights @ X[:-1])
            print("real:\t", X[-1])
            print("grad:",  grad)
            print("lr_denom", self.lr_denom)
        self.weights -= 1 / self.learning_rate / self.lr_denom * grad
        next_price = self.weights @ X[-self.window:]
        return next_price

    def predict_next_price_arima_ons(self, X):
        diff = self.weights @ X[:-1] - X[-1]
        if self.verbose:
            print("pred:\t", self.weights @ X[:-1])
            print("real:\t", X[-1])
        grad = 2 * X[:-1] * diff
        grad = grad.reshape(1, len(grad))
        self.A += grad.T @ grad
        B = (np.linalg.inv(self.A) @ grad.T).flatten()
        self.weights -= 1 / self.learning_rate / self.lr_denom * B
        next_price = self.weights @ X[-self.window:]
        return next_price

    def predict_next_price_arima_ons_simple(self, X):
        diff = self.weights @ X[:-1] - X[-1]
        grad = 2 * X[:-1] * diff
        grad = grad.reshape(1, len(grad))
        # Sherman-Morrison formula
        denom = 1 + grad @ self.A_trans @ grad.T
        numer = self.A_trans @ grad.T @ grad @ self.A_trans
        self.A_trans = self.A_trans -  numer / denom
        B = (grad @ self.A_trans).flatten()
        self.weights -= 1 / self.learning_rate / self.lr_denom * B
        next_price = self.weights @ X[-self.window:]
        return next_price

    def predict_next_price(self, X, lr_denom=1.):
        if self.lr_decrease_power is not None:
            self.lr_denom = lr_denom**self.lr_decrease_power
        else:
            self.lr_denom = 1
        if self.method == "ons":
            next_price = self.predict_next_price_arima_ons(X)
        elif self.method == "ons_simple":
            next_price = self.predict_next_price_arima_ons_simple(X)
        elif self.method == "ogd":
            next_price = self.predict_next_price_arima_ogd(X)
        else:
            print("Wrong method: ogd and ons are available")
            exit(1)
        return next_price


def anticor_expert(data, weight_o, w):
    T, N = data.shape
    weights = weight_o.copy()
    if T < 2 * w:
        return weights

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
    weights -= np.sum(transfer_ij, axis=0)
    return weights
