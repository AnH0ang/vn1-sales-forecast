from pprint import pprint

import jax
import jax.numpy as jnp
import numpy as np
import optax
import polars as pl
from tqdm import tqdm

from vn1_sales_forecast.cv import split_cv_loo
from vn1_sales_forecast.settings import PRED_PREFIX


@jax.jit
def _score(y_pred: jax.Array, y_true: jax.Array) -> jax.Array:
    mae = jnp.sum(jnp.abs(y_pred - y_true)) / jnp.sum(jnp.abs(y_true))
    bias = jnp.abs(jnp.sum(y_pred - y_true)) / jnp.sum(jnp.abs(y_true))
    score = 0.5 * mae + 0.5 * bias
    return score


@jax.jit
def _compute_loss(
    params: dict[str, jax.Array],
    X: jax.Array,
    y: jax.Array,
    alpha1: float = 0.0,
    alpha2: float = 0.1,
) -> jax.Array:
    w = params["w"]
    y_pred = X @ w
    l2_reg = jnp.mean(w**2)
    l1_reg = jnp.mean(jnp.abs(w))
    return _score(y_pred, y) + alpha2 * l2_reg + alpha1 * l1_reg


def _optimize_weights(
    xs: jax.Array, ys: jax.Array, xs_val: jax.Array | None = None, ys_val: jax.Array | None = None
) -> np.ndarray:
    params = {"w": jnp.ones(xs.shape[1]) / xs.shape[1]}

    scheduler = optax.warmup_exponential_decay_schedule(
        init_value=0.1, peak_value=0.05, warmup_steps=500, transition_steps=200, decay_rate=0.5
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1),
    )
    opt_state = tx.init(params)

    for i in range(1_200):
        if i % 100 == 0:
            r = f"step: {i:0>3}, loss: {_compute_loss(params, xs, ys):.3f}"
            if xs_val is not None and ys_val is not None:
                r += f" val: {_compute_loss(params, xs_val, ys_val):.3f}"
            print(r)

        grads = jax.grad(_compute_loss)(params, xs, ys)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    return np.array(params["w"])  # type: ignore


def _find_optimal_pred(
    train_kernel: pl.LazyFrame,
    val_kernel: pl.LazyFrame,
    total_scores: pl.LazyFrame,
    cv: bool = True,
) -> pl.LazyFrame:
    # Sort models by score
    model_list = total_scores.sort("score").collect()["model"].to_list()

    ps: list[pl.Series] = []

    KS = [None]
    CENSORED_HIGH_VARIANCE_MODELS = [
        "LGBMRegressorDirect",
        "CrostonOptimized",
        "OptimizedTheta",
        "DynamicOptimizedTheta",
        "AutoETS",
        "AutoMFLES",
        "SESOpt",
        "IMAPA",
        "KAN",
        "NHITS",
        "NHITSCustom",
        "LGBMRegressorRecursivePartitioned",
        "WindowAverage",
        "ZeroModel",
    ]

    for k in KS:
        used_models = model_list if k is None else model_list[:k]
        used_models = list(sorted(set(used_models) - set(CENSORED_HIGH_VARIANCE_MODELS)))
        pred_cols = pl.col([f"{PRED_PREFIX}{m}" for m in used_models])

        X_train = train_kernel.select(pred_cols).collect()
        y_train = train_kernel.select("sales").collect().to_series()

        X_val = val_kernel.select(pred_cols).collect()

        if cv:
            y_val = val_kernel.select("sales").collect().to_series().to_jax()
        else:
            y_val = None

        w = _optimize_weights(X_train.to_jax(), y_train.to_jax(), X_val.to_jax(), y_val)

        pprint(dict(zip(X_train.columns, map(float, w))))
        pprint(w.sum())

        p_name = f"{PRED_PREFIX}OptimizedWeights{'' if k is None else f'Top{k}'}Ensemble"
        p = np.clip(X_val.to_numpy() @ w, 0, None)
        p = pl.Series(p_name, p)
        ps.append(p)

    # Add cutoff date to the output if cv is True
    return_cols = ["id", "date"] + (["cutoff_date"] if cv else []) + ps
    return val_kernel.select(*return_cols)


def cross_validate(
    cv_pred: pl.LazyFrame,
    train: pl.LazyFrame,
    total_scores: pl.LazyFrame,
) -> pl.LazyFrame:
    kernel = cv_pred.join(train.select("id", "date", "sales"), on=["id", "date"], how="inner")
    kernel = kernel.sort("id", "date", "cutoff_date")

    preds: list[pl.LazyFrame] = []
    for train_kernel, val_kernel in tqdm(list(split_cv_loo(kernel))):
        p = _find_optimal_pred(train_kernel, val_kernel, total_scores, cv=True)
        preds.append(p)
    return pl.concat(preds)


def live_forecast(
    cv_pred: pl.LazyFrame,
    train: pl.LazyFrame,
    live_pred: pl.LazyFrame,
    total_scores: pl.LazyFrame,
) -> pl.LazyFrame:
    kernel = cv_pred.join(train.select("id", "date", "sales"), on=["id", "date"], how="inner")
    kernel = kernel.sort("id", "date", "cutoff_date")

    p = _find_optimal_pred(kernel, live_pred, total_scores, cv=False)
    return p
