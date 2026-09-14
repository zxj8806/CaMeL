
import numpy as np
import scipy
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
def compute_dci(
    true_z_train, model_z_train, true_z_test, model_z_test, return_full_importance_matrix=False
):
    _verify_inputs(true_z_train, model_z_train, true_z_test, model_z_test)
    model_z_test = model_z_test.detach().cpu().data.numpy()
    model_z_train = model_z_train.detach().cpu().data.numpy()
    true_z_test = true_z_test.detach().cpu().data.numpy()
    true_z_train = true_z_train.detach().cpu().data.numpy()
    importance_matrix, train_err, test_err = _train_dci_classifier(
        true_z_train, model_z_train, true_z_test, model_z_test
    )
    metrics = {
        "informativeness_train": train_err,
        "informativeness_test": test_err,
        "disentanglement": _compute_disentanglement(importance_matrix),
        "completeness": _compute_completeness(importance_matrix),
    }
    if return_full_importance_matrix:
        for i in range(importance_matrix.shape[0]):
            for j in range(importance_matrix.shape[1]):
                metrics[f"importance_matrix_{i}_{j}"] = importance_matrix[i, j]
    return metrics
def _verify_inputs(true_z_train, model_z_train, true_z_test, model_z_test):
    assert (
        len(true_z_train.shape)
        == len(model_z_train.shape)
        == len(true_z_test.shape)
        == len(model_z_test.shape)
        == 2
    )
    batchsize_train, dim_z_true = true_z_train.shape
    batchsize_test, dim_z_model = model_z_test.shape
    assert true_z_test.shape == (batchsize_test, dim_z_true)
    assert model_z_train.shape == (batchsize_train, dim_z_model)
def _train_dci_classifier(true_z_train, model_z_train, true_z_test, model_z_test):
    _, dim_z_true = true_z_train.shape
    _, dim_z_model = model_z_test.shape
    importance_matrix = np.zeros(shape=[dim_z_model, dim_z_true])
    train_errors, test_errors = [], []
    for i in range(dim_z_true):
        model = GradientBoostingRegressor()
        model.fit(model_z_train, true_z_train[:, i])
        importance_matrix[:, i] = np.abs(model.feature_importances_)
        train_errors.append(mean_squared_error(model.predict(model_z_train), true_z_train[:, i]))
        test_errors.append(mean_squared_error(model.predict(model_z_test), true_z_test[:, i]))
    return importance_matrix, np.mean(train_errors), np.mean(test_errors)
def _compute_disentanglement(importance_matrix, eps=1.0e-12):
    disentanglement_per_code = 1.0 - scipy.stats.entropy(
        importance_matrix.T + eps, base=importance_matrix.shape[1]
    )
    if np.abs(importance_matrix.sum()) < eps:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()
    disentanglement = np.sum(disentanglement_per_code * code_importance)
    return disentanglement
def _compute_completeness(importance_matrix, eps=1.0e-12):
    completeness_per_factor = 1.0 - scipy.stats.entropy(
        importance_matrix + eps, base=importance_matrix.shape[0]
    )
    if np.abs(importance_matrix.sum()) < eps:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    completeness = np.sum(completeness_per_factor * factor_importance)
    return completeness






