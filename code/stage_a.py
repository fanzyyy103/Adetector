# Stage A: reproduce the lecture result, then look at the confirmed insiders vs everyone else.
# Uses only data/wallet_features.csv in the repo root (no database, no passwords). In VS Code each "# %%" is a runnable cell.

# %% 0. Setup
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()   # repo root
if not (ROOT / "data").exists() and (ROOT.parent / "data").exists():                        # cell run from code/
    ROOT = ROOT.parent
DATA = ROOT / "data" / "wallet_features.csv"   # put the course CSV here (the data/ folder is git-ignored)
OUT = ROOT / "stage_a_out"; OUT.mkdir(exist_ok=True)
if not DATA.exists():
    raise SystemExit(f"Missing {DATA}. Copy wallet_features.csv from the course materials into the data/ folder.")
FEATS = ["position_anomaly_ratio", "market_notional_ratio", "avg_buy_price",
         "concentration", "wallet_age_days", "unique_markets"]
ALL = FEATS + ["trade_count", "total_notional"]
KS = (0.5, 1, 2.5, 5, 10)

# %% 1. Load, set bots aside (same rule as the lecture)
W = pd.read_csv(DATA, index_col="wallet")
W["is_bot"] = (W.unique_markets >= 500) | (W.trade_count >= 2000)
print(f"{len(W):,} wallets, {int(W.confirmed.sum())} confirmed; "
      f"bots set aside: {W.is_bot.sum()} (confirmed among them: {int(W[W.is_bot].confirmed.sum())})")
H = W[~W.is_bot].copy()
n, pos = len(H), int(H.confirmed.sum())
print(f"scored wallets: {n:,}, confirmed insiders scored: {pos}")

# %% 2. Isolation Forest (reproduces the lecture: 1 / 1 / 2 / 5 / 8 caught)
X = H[FEATS].to_numpy(float)
for c in range(X.shape[1]):
    bad = ~np.isfinite(X[:, c]); X[bad, c] = np.nanmedian(X[~bad, c])
forest = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1).fit(X)
H["if_score"] = -forest.decision_function(X)          # higher = more anomalous

def caught(score):
    """How many confirmed insiders are in the top k% of this ranking."""
    r = pd.Series(score, index=H.index).rank(ascending=False, method="first")
    return {f"top {k}%": int((r[H.confirmed] <= int(np.ceil(k / 100 * n))).sum()) for k in KS}

# %% 3. Baselines: biggest spenders, and each feature alone (direction fixed in advance by hypothesis)
HYP = {"position_anomaly_ratio": +1, "market_notional_ratio": +1, "avg_buy_price": -1,
       "concentration": +1, "wallet_age_days": -1, "unique_markets": -1,
       "trade_count": +1, "total_notional": +1}      # +1 = higher means more suspicious
rows = [{"ranking": "Isolation Forest", "AUC": roc_auc_score(H.confirmed, H.if_score), **caught(H.if_score)}]
for f in ALL:
    s = HYP[f] * H[f].fillna(H[f].median())
    rows.append({"ranking": f"{f} ({'high' if HYP[f] > 0 else 'low'} = suspicious)",
                 "AUC": roc_auc_score(H.confirmed, s), **caught(s)})
res = pd.DataFrame(rows).round(3)
res.loc[len(res)] = {"ranking": "chance", "AUC": 0.5, **{f"top {k}%": round(pos * k / 100, 2) for k in KS}}
print(res.to_string(index=False)); res.to_csv(OUT / "rankings_comparison.csv", index=False)
# AUC below 0.5 means the feature points the OTHER way from the hypothesis.

# %% 4. Insiders vs everyone else: medians and each insider's percentile within every feature
cmp = pd.DataFrame({"median_confirmed": H[H.confirmed][ALL].median(),
                    "median_others": H[~H.confirmed][ALL].median()})
cmp["ratio"] = cmp.median_confirmed / cmp.median_others.replace(0, np.nan)
print(cmp.round(3)); cmp.round(4).to_csv(OUT / "feature_medians.csv")
pct = H[ALL].rank(pct=True).loc[H.confirmed].mul(100).round(1)   # 50 = typical, 99 = top of the population
pct.insert(0, "if_rank", H.if_score.rank(ascending=False, method="first").loc[H.confirmed].astype(int))
print(pct.sort_values("if_rank").to_string()); pct.to_csv(OUT / "insider_percentiles.csv")

# %% 5. Figures
fig, axs = plt.subplots(2, 4, figsize=(16, 7))
for ax, f in zip(axs.ravel(), ALL):
    for lab, d in (("others", H[~H.confirmed]), ("confirmed", H[H.confirmed])):
        v = np.sort(np.log1p(d[f].clip(lower=0).dropna()))
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), label=lab, lw=2 if lab == "confirmed" else 1.2)
    ax.set_title(f, fontsize=10); ax.set_xlabel("log(1 + value)")
axs[0, 0].legend(); axs[0, 0].set_ylabel("cumulative share of wallets")
plt.tight_layout(); plt.savefig(OUT / "feature_ecdf.png", dpi=150); plt.show()

fig, ax = plt.subplots(figsize=(8, 4))
w = 0.2
for i, name in enumerate(["Isolation Forest", "total_notional (high = suspicious)"]):
    r = res[res.ranking == name].iloc[0]
    ax.bar(np.arange(len(KS)) + i * w, [r[f"top {k}%"] for k in KS], w, label=name)
ax.plot(np.arange(len(KS)) + w / 2, [pos * k / 100 for k in KS], "k--", label="chance")
ax.set_xticks(np.arange(len(KS)) + w / 2); ax.set_xticklabels([f"top {k}%" for k in KS])
ax.set_ylabel(f"confirmed insiders caught (of {pos})"); ax.legend()
plt.tight_layout(); plt.savefig(OUT / "caught_by_ranking.png", dpi=150); plt.show()
print("saved to", OUT.resolve())
