import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def ser(trues, preds, phi_trues=None, ph=None, t=0):
    """Squared Error for Relevance (SER)."""
    if phi_trues is None and ph is None:
        raise ValueError("You need to input either the parameter phi_trues or ph.")
    if phi_trues is None:
        phi_trues = phi(trues, ph)

    error = (trues[phi_trues >= t] - preds[phi_trues >= t]) ** 2
    error = np.nan_to_num(error)
    return np.sum(error)

def sera(trues, preds, phi_trues=None, ph=None, pl=False,
         m_name="Model", step=0.001, return_err=False, norm=False):
    """SERA metric (Squared Error for Relevance Areas)."""
    if phi_trues is None and ph is None:
        raise ValueError("You need to input either the parameter phi_trues or ph.")
    if phi_trues is None:
        phi_trues = phi(trues, ph)

    tbl = pd.DataFrame({'trues': trues, 'phi': phi_trues, 'preds': preds})
    th = np.arange(0, 1 + step, step)
    ms = tbl.columns[2:]

    errors = []
    for m in ms:
        error_m = []
        for x in th:
            mask = tbl['phi'] >= x
            if len(tbl[mask]) > 0:
                error_value = np.sum((tbl.loc[mask, 'trues'].values - tbl.loc[mask, m].values) ** 2)
            else:
                error_value = 0
            error_m.append(error_value)
        errors.append(error_m)

    errors = np.array(errors)
    if errors.ndim == 1:
        errors = np.expand_dims(errors, axis=0)

    if norm:
        errors = errors / errors[0]

    areas = np.array([[step * (errors[0, x-1] + errors[0, x]) / 2 for x in range(1, len(th))]])
    res = np.sum(areas, axis=1)

    if pl:
        plt.figure(figsize=(10, 6))
        for i, m in enumerate(ms):
            sns.lineplot(x=th, y=errors[i], label=m)
        plt.xlabel("Relevance φ(y)")
        plt.ylabel("SER")
        plt.title("SERA")
        plt.axhline(0, color="black")
        plt.legend()
        plt.show()

    if return_err:
        return {'sera': res, 'errors': errors, 'thrs': th}
    else:
        return res
