class Color:
    r: float
    g: float
    b: float

def mix(c1 : In[Color], c2 : In[Color]) -> Color:
    c_out : Color
    c_out.r = (c1.r + c2.r)/2
    c_out.g = (c1.g + c2.g)/2
    c_out.b = (c1.b + c2.b)/2
    return c_out

# Assume Diff[T], fwd_diff(f) is already defined

# def mix_loss(c1: In[Color], c2: In[Color], target: In[Color]) -> float:
#     c_mix : Color
#     dr : float
#     dg : float
#     db : float
#     c_mix = mix(c1, c2)
#     dr = c_mix.r - target.r
#     dg = c_mix.g - target.g
#     db = c_mix.b - target.b
#     return dr * dr + dg * dg + db * db

def mix_loss(r1: In[float], g1: In[float], b1: In[float],
             r2: In[float], g2: In[float], b2: In[float],
             rt: In[float], gt: In[float], bt: In[float]) -> float:
    r_mix : float = (r1 + r2) / 2
    g_mix : float = (g1 + g2) / 2
    b_mix : float = (b1 + b2) / 2
    dr : float = r_mix - rt
    dg : float = g_mix - gt
    db : float = b_mix - bt
    return dr * dr + dg * dg + db * db

fwd_mix_loss = fwd_diff(mix_loss)

def grad_mix_loss(r1: In[float], g1: In[float], b1: In[float],
             r2: In[float], g2: In[float], b2: In[float],
             rt: In[float], gt: In[float], bt: In[float],
             dr1: Out[float], dg1: Out[float], db1: Out[float], dr2: Out[float], dg2: Out[float], db2: Out[float]):

    # ∂loss/∂r1
    d_r1 : Diff[float]
    d_r1.val = r1
    d_r1.dval = 1.0
    d_g1 : Diff[float]
    d_g1.val = g1
    d_g1.dval = 0.0
    d_b1 : Diff[float]
    d_b1.val = b1
    d_b1.dval = 0.0
    d_r2 : Diff[float]
    d_r2.val = r2
    d_r2.dval = 0.0
    d_g2 : Diff[float]
    d_g2.val = g2
    d_g2.dval = 0.0
    d_b2 : Diff[float]
    d_b2.val = b2
    d_b2.dval = 0.0

    d_rt : Diff[float]
    d_rt.val = rt
    d_rt.dval = 0.0

    d_gt : Diff[float]
    d_gt.val = gt
    d_gt.dval = 0.0

    d_bt : Diff[float]
    d_bt.val = bt
    d_bt.dval = 0.0
    
    dr1 = fwd_mix_loss(d_r1, d_g1, d_b1, d_r2, d_g2, d_b2, d_rt, d_gt, d_bt).dval

    # ∂loss/∂g1
    d_r1.dval = 0.0
    d_g1.dval = 1.0
    dg1 = fwd_mix_loss(d_r1, d_g1, d_b1, d_r2, d_g2, d_b2, d_rt, d_gt, d_bt).dval

    # ∂loss/∂b1
    d_g1.dval = 0.0
    d_b1.dval = 1.0
    db1 = fwd_mix_loss(d_r1, d_g1, d_b1, d_r2, d_g2, d_b2, d_rt, d_gt, d_bt).dval

    # ∂loss/∂r2
    d_b1.dval = 0.0
    d_r2.dval = 1.0
    dr2 = fwd_mix_loss(d_r1, d_g1, d_b1, d_r2, d_g2, d_b2, d_rt, d_gt, d_bt).dval

    # ∂loss/∂g2
    d_r2.dval = 0.0
    d_g2.dval = 1.0
    dg2 = fwd_mix_loss(d_r1, d_g1, d_b1, d_r2, d_g2, d_b2, d_rt, d_gt, d_bt).dval

    # ∂loss/∂b2
    d_g2.dval = 0.0
    d_b2.dval = 1.0
    db2 = fwd_mix_loss(d_r1, d_g1, d_b1, d_r2, d_g2, d_b2, d_rt, d_gt, d_bt).dval

# def grad_mix_loss(c1: In[Color], c2: In[Color], target: In[Color], dc1: Out[Color], dc2: Out[Color]):
#     # All six partials: dr1, dg1, db1, dr2, dg2, db2
#     d_c1 : Color
#     d_c2 : Color
#     # d_c1 = Color()
#     # d_c2 = Color()
#     d_c1.r : Diff[float]
#     d_c1.r.val = c1.r
#     d_c1.r.dval = 1.0
#     d_c1.g : Diff[float]
#     d_c1.g.val = c1.g
#     d_c1.g.dval = 0.0
#     d_c1.b : Diff[float]
#     d_c1.b.val = c1.b
#     d_c1.b.dval = 0.0
#     d_c2.r : Diff[float]
#     d_c2.r.val = c2.r
#     d_c2.r.dval = 0.0
#     d_c2.g : Diff[float]
#     d_c2.g.val = c2.g
#     d_c2.g.dval = 0.0
#     d_c2.b : Diff[float]
#     d_c2.b.val = c2.b
#     d_c2.b.dval = 0.0
#     dc1.r = fwd_mix_loss(d_c1, d_c2, target).dval

#     # Partial wrt c1.g
#     d_c1.r.dval = 0.0
#     d_c1.g.dval = 1.0
#     dc1.g = fwd_mix_loss(d_c1, d_c2, target).dval

#     # Partial wrt c1.b
#     d_c1.g.dval = 0.0
#     d_c1.b.dval = 1.0
#     dc1.b = fwd_mix_loss(d_c1, d_c2, target).dval

#     # Partial wrt c2.r
#     d_c1.b.dval = 0.0
#     d_c2.r.dval = 1.0
#     dc2.r = fwd_mix_loss(d_c1, d_c2, target).dval

#     # Partial wrt c2.g
#     d_c2.r.dval = 0.0
#     d_c2.g.dval = 1.0
#     dc2.g = fwd_mix_loss(d_c1, d_c2, target).dval

#     # Partial wrt c2.b
#     d_c2.g.dval = 0.0
#     d_c2.b.dval = 1.0
#     dc2.b = fwd_mix_loss(d_c1, d_c2, target).dval