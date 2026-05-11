# TSA Project — Dossier des Workstreams

## Structure

```
TSA_Project/
├── WorkstreamA.ipynb   — Panier FX (OLS, régression glissante)
├── WorkstreamB.ipynb   — Spread interbancaire (AR, GARCH, Markov)
├── WorkstreamC.ipynb   — Nowcasting + valeur intrinsèque (CORRIGÉ)
└── README.md
```

## Corrections appliquées dans Workstream C

### Preuve 1 — Dates
Merge direct sur Date réelle. Plus de réattribution artificielle.

### Preuve 2 — Cible de validation
```python
# Ligne ajoutée :
df['IB_next'] = df['USD_IB'].shift(-1)

# Dans la boucle :
actuals.append(row['IB_next'])   # IB_publié(t+1) = taux réel de t
```

### Preuve 3 — Kalman en prédiction pure
```python
# Avant (look-ahead bias) :
x_hat = method2_pred + K * (y_t - method2_pred)
preds.append(x_hat)

# Après (nowcasting pur) :
kalman_pred = method2_pred      # stocké AVANT de voir y_t
preds.append(kalman_pred)
K = P_pred / (P_pred + R)
P = (1 - K) * P_pred            # P mis à jour, pas x_hat
```

## Performances finales (validation correcte)

| Modèle | MAE |
|--------|-----|
| Naïf 1 — Fixing seul | 0.01160 |
| Naïf 2 — Fixing + spread(t-1) | 0.00968 |
| **Markov-AR(2) + Kalman** | **0.00765** |
| EWMA optimisé | ~0.00800 |
| XGBoost | ~0.00967 |

## Ce qui n'a PAS changé
- Architecture générale identique à ta version
- AR(2) par régime
- GARCH(1,1)
- Markov 2 régimes
- XGBoost features
- Split 80/20
