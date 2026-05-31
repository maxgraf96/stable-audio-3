"""Bit-exactness reference for the SAME-S encoder port (Python side).

Usage: python verify_sames_encoder.py [T_lat]
"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mlx"))
from models.defs.same_s_encoder import load_model  # noqa: E402

NPZ = (Path(__file__).resolve().parent.parent / "mlx" / "models" / "mlx"
       / "same_s_encoder_f32.npz")


def dump(name, a):
    a = np.array(a.astype(mx.float32))
    print(f"{name} shape={list(a.shape)}")
    flat = a.flatten()
    print("  first16:", " ".join(f"{v:.6f}" for v in flat[:16]))
    print(f"  mean={flat.mean():.6f} std={flat.std():.6f} "
          f"absmax={np.abs(flat).max():.6f} n={flat.size}")


def main():
    T_lat = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    T_aud = T_lat * 16
    m = load_model(weights_path=str(NPZ), dtype=mx.float32)
    total = 512 * T_aud
    patches = mx.reshape(mx.arange(total, dtype=mx.float32), [1, 512, T_aud]) / total
    out = m(patches)
    dump("enc", out)
    print(f"SAME-S encoder, T_lat={T_lat} T_aud={T_aud}")


if __name__ == "__main__":
    main()
