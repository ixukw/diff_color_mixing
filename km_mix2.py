# Types
class Pigment:
    K: Array[float, 5]
    S: Array[float, 5]

class Color:
    r: float
    g: float
    b: float

# Reflectance and color computation
def reflectance(K: In[float], S: In[float]) -> float:
    ks : float = K / S
    return 1 + ks - sqrt(ks * ks + 2 * ks)

def saunderson(R: In[float]) -> float:
    K1: float = 0.0031
    K2: float = 0.650
    return ((1 - K1) * (1 - K2) * R) / (1 - K2 * R)

def clip(x: In[float], lo: In[float], hi: In[float]) -> float:
    return max(lo, min(x, hi))

def max(a: In[float], b: In[float]) -> float:
    c: float = 0.0
    if a > b:
        c = a
    else:
        c = b
    return c

def min(a: In[float], b: In[float]) -> float:
    c: float = 0.0
    if a < b:
        c = a
    else:
        c = b
    return c

def km_color(p: In[Pigment]) -> Color:
    X: float = 0.0
    Y: float = 0.0
    Z: float = 0.0
    YD65: float = 11619.34742175

    # OBS_XYZ : Array[Array[float,3],5]
    # OBS_XYZ[0][0] = 9.9951e-03
    # OBS_XYZ[0][1] = 0.0000e+00
    # OBS_XYZ[0][2] = 3.4983e-02
    # OBS_XYZ[1][0] = 2.2467e+01
    # OBS_XYZ[1][1] = 2.1272e+01
    # OBS_XYZ[1][2] = 1.5134e+02
    # OBS_XYZ[2][0] = 7.0520e+01
    # OBS_XYZ[2][1] = 9.9730e+01
    # OBS_XYZ[2][2] = 0.0000e+00
    # OBS_XYZ[3][0] = 1.2241e+01
    # OBS_XYZ[3][1] = 4.8369e+00
    # OBS_XYZ[3][2] = 0.0000e+00
    # OBS_XYZ[4][0] = 1.9078e-02
    # OBS_XYZ[4][1] = 6.3593e-03
    # OBS_XYZ[4][2] = 0.0000e+00

    i: int = 0
    R: float = 0.0
    while (i < 5, max_iter := 100):
        R = reflectance(p.K[i], p.S[i])
        R = saunderson(R)
        if i == 0:
            X = X + R * 9.9951e-03
            Y = Y + R * 0.0000e+00
            Z = Z + R * 3.4983e-02
        elif i == 1:
            X = X + R * 2.2467e+01
            Y = Y + R * 2.1272e+01
            Z = Z + R * 1.5134e+02
        elif i == 2:
            X = X + R * 7.0520e+01
            Y = Y + R * 9.9730e+01
            Z = Z + R * 0.0000e+00
        elif i == 3:
            X = X + R * 1.2241e+01
            Y = Y + R * 4.8369e+00
            Z = Z + R * 0.0000e+00
        elif i == 4:
            X = X + R * 1.9078e-02
            Y = Y + R * 6.3593e-03
            Z = Z + R * 0.0000e+00
        i = i + 1
    
    X = X/YD65
    Y = Y/YD65
    Z = Z/YD65

    # XYZ_TO_RGB = [[3.2404542, -1.5371385, -0.4985314],
    #           [-0.9692660,  1.8760108,  0.0415560],
    #           [0.0556434, -0.2040259,  1.0572252]]

    # r = XYZ_TO_RGB[0][0] * X + XYZ_TO_RGB[0][1] * Y + XYZ_TO_RGB[0][2] * Z
    # g = XYZ_TO_RGB[1][0] * X + XYZ_TO_RGB[1][1] * Y + XYZ_TO_RGB[1][2] * Z
    # b = XYZ_TO_RGB[2][0] * X + XYZ_TO_RGB[2][1] * Y + XYZ_TO_RGB[2][2] * Z
    c: Color
    r: float = 3.2404542 * X +  -1.5371385 * Y + -0.4985314 * Z
    g: float = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    b: float = 0.0556434 * X + -0.2040259 * Y + 1.0572252 * Z

    c.r = clip(r, 0.001, 1.0)
    c.g = clip(g, 0.001, 1.0)
    c.b = clip(b, 0.001, 1.0)
    return c

# Linear mixing of two pigments
def km_mix(p1: In[Pigment], p2: In[Pigment], t: In[float]) -> Pigment:
    out: Pigment
    i: int = 0
    while (i < 5, max_iter := 100):
        out.K[i] = p1.K[i] * (1 - t) + p2.K[i] * t
        out.S[i] = p1.S[i] * (1 - t) + p2.S[i] * t
        i = i + 1
    return out

# RGB-space loss vs. target
def km_mix_loss(p1: In[Pigment], p2: In[Pigment], t: In[float], target: In[Color]) -> float:
    mix: Pigment = km_mix(p1, p2, t)
    c: Color = km_color(mix)
    dr: float = c.r - target.r
    dg: float = c.g - target.g
    db: float = c.b - target.b
    return dr * dr + dg * dg + db * db

fwd_km_mix_loss = fwd_diff(km_mix_loss)

def grad_km_mix_loss(p1: In[Pigment], p2: In[Pigment], t: In[float], target: In[Color], dp1: Out[Pigment], dp2: Out[Pigment]):
    d_p1: Diff[Pigment]
    d_p2: Diff[Pigment]
    d_t: Diff[float]
    d_target: Diff[Color]
    i: int = 0
    
    # init diffs
    i = 0
    while (i < 5, max_iter := 100):
        d_p1.K[i].val = p1.K[i]
        d_p1.S[i].val = p1.S[i]
        d_p1.K[i].dval = 0.0 
        d_p1.S[i].dval = 0.0
        i = i + 1
    i = 0
    while (i < 5, max_iter := 100):
        d_p2.K[i].val = p2.K[i]
        d_p2.S[i].val = p2.S[i]
        d_p2.K[i].dval = 0.0
        d_p2.S[i].dval = 0.0
        i = i + 1

    d_target.r.val = target.r
    d_target.g.val = target.g
    d_target.b.val = target.b
    d_target.r.dval = 0.0
    d_target.g.dval = 0.0
    d_target.b.dval = 0.0
    d_t.val = t
    d_t.dval = 0.0

    # Compute gradients for p1.K
    i = 0
    while (i < 5, max_iter := 100):
        d_p1.K[i].dval = 1.0
        d_p1.S[i].dval = 0.0
        d_p2.K[i].dval = 0.0
        d_p2.S[i].dval = 0.0
        dp1.K[i] = fwd_km_mix_loss(d_p1, d_p2, d_t, d_target).dval
        i = i + 1
    # Compute gradients for p1.S
    i = 0
    while (i < 5, max_iter := 100):
        d_p1.K[i].dval = 0.0
        d_p1.S[i].dval = 1.0
        d_p2.K[i].dval = 0.0
        d_p2.S[i].dval = 0.0
        dp1.S[i] = fwd_km_mix_loss(d_p1, d_p2, d_t, d_target).dval
        i = i + 1
    # Compute gradients for p2.K
    i = 0
    while (i < 5, max_iter := 100):
        d_p1.K[i].dval = 0.0
        d_p1.S[i].dval = 0.0
        d_p2.K[i].dval = 1.0
        d_p2.S[i].dval = 0.0
        dp2.K[i] = fwd_km_mix_loss(d_p1, d_p2, d_t, d_target).dval
        i = i + 1
    # Compute gradients for p2.S
    i = 0
    while (i < 5, max_iter := 100):
        d_p1.K[i].dval = 0.0
        d_p1.S[i].dval = 0.0
        d_p2.K[i].dval = 0.0
        d_p2.S[i].dval = 1.0
        dp2.S[i] = fwd_km_mix_loss(d_p1, d_p2, d_t, d_target).dval
        i = i + 1