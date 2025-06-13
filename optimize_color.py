import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler
import ctypes

# Setup paths
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# Compile C code
with open('rgb_mix.py') as f:
    _, lib = compiler.compile(f.read(),
                              target='c',
                              output_filename='_code/rgb_mix')

f = lib.mix_loss
grad_f = lib.grad_mix_loss

# Target color
rt, gt, bt = 0.1, 0.0, 0.9

# Initialize colors
r1, g1, b1 = 0.0, 0.9, 0.0
r2, g2, b2 = 0.9, 0.0, 0.0
lr = 0.1

traj_rgb = []

print(f"Start c1: r={r1:.4f}, g={g1:.4f}, b={b1:.4f}")
print(f"Start c2: r={r2:.4f}, g={g2:.4f}, b={b2:.4f}")

for step in range(300):
    r_mix = (r1 + r2) / 2
    g_mix = (g1 + g2) / 2
    b_mix = (b1 + b2) / 2
    traj_rgb.append([r_mix, g_mix, b_mix])

    dr1 = ctypes.c_float(0); dg1 = ctypes.c_float(0); db1 = ctypes.c_float(0)
    dr2 = ctypes.c_float(0); dg2 = ctypes.c_float(0); db2 = ctypes.c_float(0)

    grad_f(
        ctypes.c_float(r1), ctypes.c_float(g1), ctypes.c_float(b1),
        ctypes.c_float(r2), ctypes.c_float(g2), ctypes.c_float(b2),
        ctypes.c_float(rt), ctypes.c_float(gt), ctypes.c_float(bt),
        ctypes.byref(dr1), ctypes.byref(dg1), ctypes.byref(db1),
        ctypes.byref(dr2), ctypes.byref(dg2), ctypes.byref(db2)
    )

    r1 -= lr * dr1.value; g1 -= lr * dg1.value; b1 -= lr * db1.value
    r2 -= lr * dr2.value; g2 -= lr * dg2.value; b2 -= lr * db2.value

    r1 = np.clip(r1, 0, 1); g1 = np.clip(g1, 0, 1); b1 = np.clip(b1, 0, 1)
    r2 = np.clip(r2, 0, 1); g2 = np.clip(g2, 0, 1); b2 = np.clip(b2, 0, 1)

# Convert to numpy
traj_rgb = np.array(traj_rgb)

# Final print
print(f"Final c1: r={r1:.4f}, g={g1:.4f}, b={b1:.4f}")
print(f"Final c2: r={r2:.4f}, g={g2:.4f}, b={b2:.4f}")
print(f"Final mixed: r={(r1 + r2)/2:.4f}, g={(g1 + g2)/2:.4f}, b={(b1 + b2)/2:.4f}")
print(f"Target color: r={rt:.4f}, g={gt:.4f}, b={bt:.4f}")

# Plot in 3D with RGB color visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create segments and colors
points = traj_rgb.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
colors = traj_rgb[:-1]

lc = Line3DCollection(segments, colors=colors, linewidths=2)
ax.add_collection3d(lc)

# Plot target color
ax.scatter(rt, gt, bt, color=(rt, gt, bt), s=100, edgecolors='black', label='Target Color')

# Plot final color
rf, gf, bf = traj_rgb[-1]
ax.scatter(rf, gf, bf, color=(rf, gf, bf), s=100, edgecolors='gray', label='Final Mixed Color')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
ax.set_title('Gradient Descent in RGB Space (Color Encoded)')
ax.legend()
plt.tight_layout()
plt.show()
