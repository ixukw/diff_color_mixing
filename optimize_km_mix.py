import matplotlib.pyplot as plt
import numpy as np
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler
import ctypes
import math

with open('project/km_mix2.py') as f:
    structs, lib = compiler.compile(f.read(),
                            target = 'c',
                            output_filename = 'project/_code/km_mix2')

km_mix_loss = lib.km_mix_loss
grad_km_mix_loss = lib.grad_km_mix_loss

Pigment = structs['Pigment']
Color = structs['Color']

# Constants
K1 = 0.0031
K2 = 0.650
YD65 = 11619.34742175

# Python versions of the functions
def reflectance(K, S):
    ks = K / S
    return 1 + ks - math.sqrt(ks * ks + 2 * ks)

def saunderson(R):
    return ((1 - K1) * (1 - K2) * R) / (1 - K2 * R)

def km_color_py(p):
    X = 0.0
    Y = 0.0
    Z = 0.0
    
    # OBS_XYZ values for 5 wavelengths
    OBS_XYZ = [
        [9.9951e-03, 0.0000e+00, 3.4983e-02],
        [2.2467e+01, 2.1272e+01, 1.5134e+02],
        [7.0520e+01, 9.9730e+01, 0.0000e+00],
        [1.2241e+01, 4.8369e+00, 0.0000e+00],
        [1.9078e-02, 6.3593e-03, 0.0000e+00]
    ]
    
    for i in range(5):
        R = reflectance(p.K[i], p.S[i])
        R = saunderson(R)
        X += R * OBS_XYZ[i][0]
        Y += R * OBS_XYZ[i][1]
        Z += R * OBS_XYZ[i][2]
    
    X /= YD65
    Y /= YD65
    Z /= YD65
    
    # XYZ to RGB conversion
    r = 3.2404542 * X + -1.5371385 * Y + -0.4985314 * Z
    g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    b = 0.0556434 * X + -0.2040259 * Y + 1.0572252 * Z
    
    # Clip values
    r = max(0.001, min(1.0, r))
    g = max(0.001, min(1.0, g))
    b = max(0.001, min(1.0, b))
    
    return Color(r=r, g=g, b=b)

def km_mix_py(p1, p2, t):
    out = Pigment()
    out.K = (ctypes.c_float * 5)(*[p1.K[i] * (1 - t) + p2.K[i] * t for i in range(5)])
    out.S = (ctypes.c_float * 5)(*[p1.S[i] * (1 - t) + p2.S[i] * t for i in range(5)])
    return out

def km_mix_loss_py(p1, p2, t, target):
    mix = km_mix_py(p1, p2, t)
    c = km_color_py(mix)
    dr = c.r - target.r
    dg = c.g - target.g
    db = c.b - target.b
    return dr * dr + dg * dg + db * db

# Create two pigments with different K/S values
def create_pigment(K_values, S_values):
    K_array = (ctypes.c_float * len(K_values))(*K_values)
    S_array = (ctypes.c_float * len(S_values))(*S_values)
    return Pigment(K=K_array, S=S_array)

# Create target color
def create_color(r, g, b):
    return Color(r=r,g=g,b=b)

# Initialize pigments with some test values
p1_K = [0.1, 0.1, 0.1, 0.1, 0.1]  # Example K values for first pigment
p1_S = [0.5, 0.5, 0.5, 0.5, 0.5]  # Example S values for first pigment
p2_K = [0.2, 0.2, 0.2, 0.2, 0.2]  # Example K values for second pigment
p2_S = [0.3, 0.3, 0.3, 0.3, 0.3]  # Example S values for second pigment

# p1 = create_pigment(p1_K, p1_S)
# p2 = create_pigment(p2_K, p2_S)

# Target color (e.g., a shade of purple)
target = create_color(0.5, 0.0, 0.5)

# Gradient descent optimization
traj_colors = []
traj_loss = []
lr = 0.01
t = 0.5

for step in range(300):
    # Create output pigment structures for gradients
    dp1 = np.zeros([2,5])#create_pigment([0]*5, [0]*5)
    dp2 = np.zeros([2,5])#create_pigment([0]*5, [0]*5)
    
    # Compute gradients
    grad_km_mix_loss(
        create_pigment(p1_K, p1_S),
        create_pigment(p2_K, p2_S),
        ctypes.c_float(t),
        target,
        dp1.ctypes.data_as(ctypes.POINTER(structs['Pigment'])),
        dp2.ctypes.data_as(ctypes.POINTER(structs['Pigment']))
    )
    
    # Update parameters using the computed gradients
    for i in range(5):
        p1_K[i] -= max(0.0, min(1.0, lr * dp1[0,i]))
        p1_S[i] -= max(0.0, min(1.0, lr * dp1[0,i]))
        p2_K[i] -= max(0.0, min(1.0, lr * dp2[1,i]))
        p2_S[i] -= max(0.0, min(1.0, lr * dp2[1,i]))

    # Store current state using Python functions
    mix = km_mix_py(create_pigment(p1_K, p1_S), create_pigment(p2_K, p2_S), t)
    current_color = km_color_py(mix)
    traj_colors.append([current_color.r, current_color.g, current_color.b])
    traj_loss.append(km_mix_loss_py(create_pigment(p1_K, p1_S), create_pigment(p2_K, p2_S), t, target))

    # if step % 100 == 0:  # Print less frequently
    print(f"Step {step}:")
    print(f"p1 K: {[f'{x:.3f}' for x in p1_K]}")
    # print(f"p1 S: {[f'{x:.3f}' for x in p1.S]}")
    # print(f"p2 K: {[f'{x:.3f}' for x in p2.K]}")
    # print(f"p2 S: {[f'{x:.3f}' for x in p2.S]}")
    print(f"Color: ({current_color.r:.3f}, {current_color.g:.3f}, {current_color.b:.3f})")
    print(f"Loss: {traj_loss[-1]:.6f}\n")

traj_colors = np.array(traj_colors)
traj_loss = np.array(traj_loss)
iterations = np.arange(len(traj_loss))

print('Final loss:', traj_loss[-1])
print('Now visualizing...')

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot loss with color gradient
for i in range(len(iterations)-1):
    ax1.plot(iterations[i:i+2], traj_loss[i:i+2], 
             color=np.clip(traj_colors[i]*100, 0, 1), 
             linewidth=2)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')
ax1.set_title('Optimization Progress')
ax1.grid(True)

# Plot color evolution
color_patches = np.zeros((len(traj_colors), 1, 3))
color_patches[:, 0, :] = traj_colors
ax2.imshow(color_patches, aspect='auto')
ax2.set_xticks([])
ax2.set_yticks([0, len(traj_colors)-1])
ax2.set_yticklabels(['Start', 'End'])
ax2.set_title('Color Evolution During Optimization')

# Add target color reference
target_patch = np.array([[target.r, target.g, target.b]])
ax2.imshow(target_patch, aspect='auto', extent=[-0.5, 0.5, -0.5, 0.5], 
           transform=ax2.transData, clip_on=False)

plt.tight_layout()
plt.show()

# Create a separate figure for the final color comparison
fig, ax = plt.subplots(figsize=(6, 4))

# Plot final mixed color using Python functions
final_mix = km_mix_py(p1, p2, t)
final_color = km_color_py(final_mix)

# Create color patches
target_color = np.array([[target.r, target.g, target.b]])
final_color_array = np.array([[final_color.r, final_color.g, final_color.b]])

# Display colors
ax.imshow(np.vstack([target_color, final_color_array]))
ax.set_xticks([])
ax.set_yticks([0, 1])
ax.set_yticklabels(['Target', 'Final'])
ax.set_title('Target vs Final Color')

# Add RGB values as text
ax.text(0.5, 0.1, f'RGB: ({target.r:.2f}, {target.g:.2f}, {target.b:.2f})', 
        ha='center', transform=ax.transAxes)
ax.text(0.5, 0.6, f'RGB: ({final_color.r:.2f}, {final_color.g:.2f}, {final_color.b:.2f})', 
        ha='center', transform=ax.transAxes)

plt.tight_layout()
plt.show() 