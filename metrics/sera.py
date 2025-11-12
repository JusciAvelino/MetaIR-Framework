import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ImbalancedLearningRegression as iblr

# ====================================================
# SER - Squared Error for Relevance
# ====================================================
def ser(trues, preds, phi_trues=None, ph=None, t=0):
    """Squared Error for Relevance (SER)."""
    if phi_trues is None and ph is None:
        # Calcula o controle de relevância automaticamente
        ph = phi_control(trues)
        phi_trues = phi(trues, ph)

    error = (trues[phi_trues >= t] - preds[phi_trues >= t]) ** 2
    error = np.nan_to_num(error)
    return np.sum(error)


# ====================================================
# SERA - Squared Error for Relevance Areas
# ====================================================
def sera(trues, preds, phi_trues=None, ph=None, pl=False,
         m_name="Model", step=0.001, return_err=False, norm=False):
    """SERA metric (Squared Error for Relevance Areas).

    Parameters
    ----------
    trues : array-like
        True target values.
    preds : array-like
        Predicted values.
    phi_trues : array-like, optional
        Relevance scores for the true values.
    ph : object, optional
        phi.control() output used for relevance calculation.
    pl : bool, optional
        Whether to plot the SER curve.
    step : float, optional
        Step size for threshold integration.
    norm : bool, optional
        Whether to normalize the error curve.
    """
    if phi_trues is None and ph is None:
        # Calcula o controle e relevância automaticamente via IBLR
        ph = phi_control(trues)
        phi_trues = phi(trues, ph)

    tbl = pd.DataFrame({'trues': trues, 'phi': phi_trues, 'preds': preds})
    th = np.arange(0, 1 + step, step)

    errors = []
    for x in th:
        mask = tbl['phi'] >= x
        if len(tbl[mask]) > 0:
            error_value = np.sum((tbl.loc[mask, 'trues'] - tbl.loc[mask, 'preds']) ** 2)
        else:
            error_value = 0
        errors.append(error_value)

    errors = np.array(errors)
    if norm:
        errors = errors / errors[0]

    # Cálculo da área sob a curva
    area = np.sum(step * (errors[:-1] + errors[1:]) / 2)

    if pl:
        plt.figure(figsize=(8, 5))
        sns.lineplot(x=th, y=errors)
        plt.xlabel("Relevance φ(y)")
        plt.ylabel("SER")
        plt.title(f"SERA - {m_name}")
        plt.show()

    if return_err:
        return {'sera': area, 'errors': errors, 'thrs': th}
    else:
        return area
