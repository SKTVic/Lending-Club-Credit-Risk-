"""Credit risk functions used by credit_risk_analysis.ipynb."""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FIG = ROOT / "figures"

# Chart colours
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------- 1. PD model
PD_FORMULA = ("default ~ fico + dti + log_income + log_amount + emp_length + emp_missing"
              " + revol_util + inq_last_6mths + delinq_2yrs + history_years"
              " + C(home_ownership) + C(verification_status) + C(purpose)")


def fit_pd_model(dev):
    """Logistic regression of default on borrower traits (application data only)."""
    return smf.logit(PD_FORMULA, data=dev).fit(disp=0, maxiter=200)


def odds_ratios(model):
    ci = model.conf_int()
    t = pd.DataFrame({"odds_ratio": np.exp(model.params), "low_95": np.exp(ci[0]),
                      "high_95": np.exp(ci[1]), "p_value": model.pvalues})
    return t.drop(index="Intercept").round(4)


# ---------------------------------------------------------------- 2. EAD and LGD
def add_loss_columns(d):
    """EAD, net recovery, LGD and realised loss (EAD and LGD only matter for defaulted loans)."""
    d = d.copy()
    d["ead"] = (d["funded_amnt"] - d["total_rec_prncp"]).clip(lower=0)
    d["ead_pct"] = d["ead"] / d["funded_amnt"]
    d["net_recovery"] = (d["recoveries"] - d["collection_recovery_fee"]).clip(lower=0)
    d["lgd"] = (1 - d["net_recovery"] / d["ead"].where(d["ead"] > 0)).clip(0, 1)
    d["actual_loss"] = np.where(d["default"] == 1,
                                (d["ead"] - d["net_recovery"]).clip(lower=0), 0.0)
    d["fico_band"] = pd.cut(d["fico"], [0, 680, 700, 720, 750, 900],
                            labels=["<680", "680-699", "700-719", "720-749", "750+"], right=False)
    return d


LGD_FORMULA = ("lgd ~ fico + dti + log_income + log_amount + int_rate + revol_util"
               " + C(home_ownership) + C(verification_status) + C(purpose)")


def fit_lgd_models(dev_def):
    """Two LGD models on defaulted dev loans: grade means, and a linear regression."""
    dev_def = dev_def.dropna(subset=["lgd"])
    return dev_def.groupby("grade")["lgd"].mean(), smf.ols(LGD_FORMULA, data=dev_def).fit()


def predict_lgd(model_name, seg, reg, d):
    if model_name == "segment":
        return d["grade"].map(seg).fillna(seg.mean()).to_numpy()
    return np.clip(reg.predict(d).to_numpy(), 0, 1)


def error_table(y, preds):
    rows = []
    for name, p in preds.items():
        e = np.asarray(p) - np.asarray(y)
        rows.append({"model": name, "MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
                     "bias (pred - actual)": e.mean()})
    return pd.DataFrame(rows).round(4)


# ---------------------------------------------------------------- 3. Default timing (IFRS 9)
def months_between(a, b):
    return (b.dt.year - a.dt.year) * 12 + (b.dt.month - a.dt.month)


def default_month(d):
    """Month the loan reached 90 days past due = month of last payment + 3."""
    return (months_between(d["issue_d"], d["last_pymnt_d"]).fillna(0) + 3).clip(1, 36)


def timing_curve(defaulted):
    """Share of lifetime defaults that have happened by each month 1..36."""
    counts = default_month(defaulted).value_counts().reindex(range(1, 37), fill_value=0)
    return counts.cumsum() / counts.sum()


def scheduled_balance(funded, rate_pct, month, term=36):
    """Amortising balance left after `month` payments (no prepayment)."""
    r = np.asarray(rate_pct) / 1200
    return np.asarray(funded) * ((1 + r) ** term - (1 + r) ** month) / ((1 + r) ** term - 1)


# ---------------------------------------------------------------- 4. Validation
def risk_deciles(p, n=10):
    return pd.qcut(pd.Series(np.asarray(p)).rank(method="first"), n,
                   labels=range(1, n + 1)).astype(int).to_numpy()


def validation_summary(y, p):
    auc = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    return {"AUC": round(auc, 4), "Gini": round(2 * auc - 1, 4), "KS": round(float(np.max(tpr - fpr)), 4)}


def psi(expected, actual, bins=10):
    """Population stability index; bins are deciles of the expected (dev) distribution."""
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-6, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-6, None)
    table = pd.DataFrame({"bin": range(1, bins + 1), "expected_share": e, "actual_share": a,
                          "psi_part": (a - e) * np.log(a / e)})
    return float(table["psi_part"].sum()), table.round(5)


# ---------------------------------------------------------------- 5. Vasicek portfolio model
def conditional_pd(pd_, rho, z):
    """PD given the economy factor z (z < 0 = bad economy). One-factor Gaussian (Vasicek)."""
    return norm.cdf((norm.ppf(pd_) - np.sqrt(rho) * z) / np.sqrt(1 - rho))


def loss_given_z(pd_, lgd, ead, rho, zgrid, chunk=200):
    """Portfolio loss at each z (large-portfolio limit, so no per-loan simulation)."""
    pd_ = np.clip(np.asarray(pd_, float), 1e-6, 0.999)
    w = np.asarray(lgd, float) * np.asarray(ead, float)
    rho = np.broadcast_to(np.asarray(rho, float), pd_.shape)
    out = np.empty(len(zgrid))
    for i in range(0, len(zgrid), chunk):
        z = zgrid[i:i + chunk][:, None]
        out[i:i + chunk] = conditional_pd(pd_[None, :], rho[None, :], z) @ w
    return out


def simulate_losses(pd_, lgd, ead, rho, n_sims=1_000_000, seed=2024):
    """Draw n_sims economy scenarios and read the portfolio loss off a fine z grid."""
    zgrid = np.linspace(-6, 6, 2401)
    curve = loss_given_z(pd_, lgd, ead, rho, zgrid)
    return np.interp(np.random.default_rng(seed).standard_normal(n_sims), zgrid, curve)


def var_es(losses, q):
    var = float(np.quantile(losses, q))
    return var, float(losses[losses >= var].mean())


def risk_report(pd_, lgd, ead, rho, label="", qs=(0.99, 0.999)):
    losses = simulate_losses(pd_, lgd, ead, rho)
    row = {"scenario": label,
           "EL": float(np.sum(np.asarray(pd_) * np.asarray(lgd) * np.asarray(ead))),
           "mean simulated loss": float(losses.mean())}
    for q in qs:
        row[f"VaR {q:.1%}"], row[f"ES {q:.1%}"] = var_es(losses, q)
    return row, losses


# ---------------------------------------------------------------- 6. Basel IRB (other retail)
PD_FLOOR = 0.0005   # 5 bp retail PD floor (Basel III / OSFI CAR)


def irb_retail_correlation(pd_):
    w = (1 - np.exp(-35 * pd_)) / (1 - np.exp(-35))
    return 0.03 * w + 0.16 * (1 - w)


def irb_capital_k(pd_, lgd):
    """Capital per $ of EAD for 'other retail': 99.9% unexpected loss, 1 year, no maturity adjustment."""
    pd_ = np.clip(pd_, PD_FLOOR, 0.999)
    R = irb_retail_correlation(pd_)
    return lgd * (norm.cdf((norm.ppf(pd_) + np.sqrt(R) * norm.ppf(0.999)) / np.sqrt(1 - R)) - pd_)


# ---------------------------------------------------------------- charts
def style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
