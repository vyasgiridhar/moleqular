#!/usr/bin/env -S uv run --with torch --with coremltools --with numpy
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "coremltools", "numpy"]
# ///
"""
Generate a CoreML neural network that approximates the LJ force F/r.

Input: r² (batch of scalars)
Output: F/r (batch of scalars)

The model runs on Apple Neural Engine (38 TOPS on M4).
Usage: uv run tools/gen_ane_model.py
"""

import torch
import torch.nn as nn
import numpy as np
import coremltools as ct

# --- Generate training data from analytical LJ force ---
N_TRAIN = 100000
r2 = np.linspace(0.64, 6.25, N_TRAIN, dtype=np.float32)
inv_r2 = 1.0 / r2
inv_r6 = inv_r2 ** 3
inv_r12 = inv_r6 ** 2
f_over_r = (24.0 * inv_r2 * (2.0 * inv_r12 - inv_r6)).astype(np.float32)

# Normalize for better training
r2_min, r2_max = r2.min(), r2.max()
f_min, f_max = f_over_r.min(), f_over_r.max()

r2_norm = (r2 - r2_min) / (r2_max - r2_min)          # [0, 1]
f_norm = (f_over_r - f_min) / (f_max - f_min + 1e-8)  # [0, 1]

# --- Model: small MLP ---
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.SiLU(),
    nn.Linear(64, 64),
    nn.SiLU(),
    nn.Linear(64, 64),
    nn.SiLU(),
    nn.Linear(64, 1),
)

# --- Train ---
X = torch.tensor(r2_norm).unsqueeze(1)
Y = torch.tensor(f_norm).unsqueeze(1)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)

print("Training ANE LJ model...")
for epoch in range(5000):
    pred = model(X)
    loss = ((pred - Y) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    if epoch % 500 == 0:
        print(f"  epoch {epoch:5d}  loss={loss.item():.6e}")

print(f"  final loss={loss.item():.6e}")

# --- Evaluate accuracy ---
model.eval()
with torch.no_grad():
    pred_np = model(X).numpy().flatten()
    pred_denorm = pred_np * (f_max - f_min + 1e-8) + f_min
    rel_err = np.abs(pred_denorm - f_over_r) / (np.abs(f_over_r) + 1e-10)
    print(f"  max relative error: {rel_err.max():.4f}")
    print(f"  mean relative error: {rel_err.mean():.6f}")
    print(f"  F/r range: [{f_over_r.min():.2f}, {f_over_r.max():.2f}]")

# --- Export to CoreML ---
print("Exporting to CoreML...")
example = torch.randn(1, 1)
traced = torch.jit.trace(model, example)

ct_model = ct.convert(
    traced,
    inputs=[ct.TensorType(name="r2_norm", shape=ct.EnumeratedShapes(shapes=[[1, 1], [256, 1], [1024, 1], [4096, 1], [16384, 1], [65536, 1]]))],
    outputs=[ct.TensorType(name="f_over_r_norm")],
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.macOS14,
)

# Store normalization constants as metadata
ct_model.user_defined_metadata["r2_min"] = str(r2_min)
ct_model.user_defined_metadata["r2_max"] = str(r2_max)
ct_model.user_defined_metadata["f_min"] = str(f_min)
ct_model.user_defined_metadata["f_max"] = str(f_max)

ct_model.save("ane_lj.mlpackage")
print("Saved ane_lj.mlpackage")
print(f"Normalization: r2 in [{r2_min}, {r2_max}] -> [0,1], F/r in [{f_min:.2f}, {f_max:.2f}] -> [0,1]")
