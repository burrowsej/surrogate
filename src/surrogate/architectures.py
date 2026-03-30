"""Factory functions that build dgpsi GP and DGP models."""

from __future__ import annotations

import logging

import numpy as np
from dgpsi.dgp import dgp
from dgpsi.emulation import emulator
from dgpsi.gp import gp
from dgpsi.kernel_class import combine, kernel

logger = logging.getLogger(__name__)

KERNEL_MAP = {
    "matern25": "matern2.5",
    "matern2.5": "matern2.5",
    "sexp": "sexp",
    "se": "sexp",
    "sqexp": "sexp",
}


def _resolve_kernel_name(name: str) -> str:
    key = name.lower().replace(" ", "").replace("-", "")
    if key not in KERNEL_MAP:
        raise ValueError(f"Unknown kernel {name!r}. Choose from: {list(KERNEL_MAP)}")
    return KERNEL_MAP[key]


def build_gp(
    X: np.ndarray,
    Y: np.ndarray,
    kernel_name: str = "matern25",
) -> gp:
    """Build and train a single-output dgpsi GP.

    Args:
        X: Input array of shape ``(n, d)``.
        Y: Output array of shape ``(n, 1)``.
        kernel_name: Kernel name (e.g. ``"matern25"``, ``"sexp"``).

    Returns:
        Trained ``dgpsi.gp`` model.
    """
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Y.shape[1] != 1:
        raise ValueError("build_gp only supports single-output. Use build_dgp for multi-output.")

    d = X.shape[1]
    kern_name = _resolve_kernel_name(kernel_name)
    kern = kernel(
        length=np.ones(d),
        name=kern_name,
        scale_est=True,
        nugget_est=True,
        prior_name="ref",
    )
    model = gp(X, Y, kern)
    model.train()
    return model


def build_dgp(
    X: np.ndarray,
    Y: np.ndarray,
    depth: int = 2,
    kernel_name: str = "matern25",
    n_iter: int = 500,
    ess_burn: int = 10,
    parallel: bool = True,
    n_imputations: int = 10,
) -> emulator:
    """Build, train and wrap a dgpsi DGP as an emulator.

    Args:
        X: Input array of shape ``(n, d)``.
        Y: Output array of shape ``(n, k)``  -- k output dimensions.
        depth: Number of DGP layers (minimum 2).
        kernel_name: Kernel name (e.g. ``"matern25"``, ``"sexp"``).
        n_iter: SEM iterations for training.
        ess_burn: ESS burn-in per SEM iteration.
        parallel: Use parallel training (ptrain).
        n_imputations: Number of imputations for the emulator.

    Returns:
        Tuple of ``(emulator, dgp)``  -- ready-to-predict emulator wrapping
        the trained DGP, and the raw DGP object.
    """
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if depth < 2:
        raise ValueError("DGP depth must be >= 2.")

    d = X.shape[1]
    k = Y.shape[1]
    kern_name = _resolve_kernel_name(kernel_name)

    # Build layer structure
    # Layer 1: D GP nodes (one per input dim), each taking one input dimension
    layer1 = []
    for i in range(d):
        layer1.append(
            kernel(
                length=np.array([1.0]),
                name=kern_name,
                scale_est=True,
                nugget_est=True,
                prior_name="ref",
                input_dim=np.array([i]),
            )
        )

    # Optional intermediate layers (depth > 2)
    intermediate_layers = []
    for _ in range(depth - 2):
        n_nodes = d  # keep same width
        layer = []
        for j in range(n_nodes):
            layer.append(
                kernel(
                    length=np.array([1.0]),
                    name=kern_name,
                    scale_est=True,
                    nugget_est=True,
                    prior_name="ref",
                )
            )
        intermediate_layers.append(layer)

    # Final layer: K GP nodes (one per output), with global input connections
    layer_final = []
    for j in range(k):
        layer_final.append(
            kernel(
                length=np.array([1.0]),
                name=kern_name,
                scale_est=True,
                nugget_est=True,
                prior_name="ref",
                connect=np.arange(d),
            )
        )

    all_layers = [layer1, *intermediate_layers, layer_final]
    all_layer = combine(*all_layers)

    model = dgp(X, Y, all_layer)

    logger.info("Training DGP with %d layers, %d iterations...", depth, n_iter)
    if parallel:
        model.ptrain(N=n_iter, ess_burn=ess_burn, disable=True)
    else:
        model.train(N=n_iter, ess_burn=ess_burn, disable=True)

    trained_structure = model.estimate()
    emu = emulator(trained_structure, N=n_imputations)

    return emu, model
