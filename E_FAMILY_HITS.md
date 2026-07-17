# E-Family CF Hits — Confirmed

Two confirmed continued fraction representations of e-related constants,
found in the 6F5 companion matrix family (dim=6, z=1/1).

---

## Hit 1: L = e - 2

| Property | Value |
|---|---|
| **Limit** | e - 2 = 0.7182818284590452353602874713526624977572... |
| **PSLQ** | [-1, -2, 1] on [L, 1, e] → -L - 2 + e = 0 |
| **Verification** | 2000 digits, zero residual |
| **Convergence** | Polynomial (δ_blind ≈ +0.0005 at N=2000) |
| **GID** | 576460752311020401 |

### Parameters
```
shift = [-1, -1,  0, -1, -1,  1, -2, -2, -1, -2,  1]
dir   = [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1]
z     = 1/1
dim   = 6
box   = 6
```

### Advancing root
g₄(n) = n + 1 (denominator root, position 10 in shift vector)

### Recurrence
a_{n+6} = c₃(n)·a_{n+3} + c₄(n)·a_{n+4} + c₅(n)·a_{n+5}

Coefficients are polynomial in n (degree 1).

### Roots at n=1
f = [0, 0, 1, 0, 0, 2],  g = [0, 0, 1, 0, 2]

### Roots at n=10
f = [0, 0, 1, 0, 0, 11], g = [0, 0, 1, 0, 11]

### Discovery
Found in z=±1 overnight sweep v1 (PID 28944), confirmed via deep PSLQ at 500 dps.

---

## Hit 2: L = e - 3/2

| Property | Value |
|---|---|
| **Limit** | e - 3/2 = 1.2182818284590452353602874713526624977572... |
| **PSLQ** | [2, 3, -2] on [L, 1, e] → 2L + 3 - 2e = 0 |
| **Verification** | 12,678 digits, zero residual |
| **Convergence** | Polynomial (δ_blind ≈ -0.0005 at N=2000) |
| **Convergent pair** | (i=5, j=4) — NOT the blind-delta-optimal pair |

### Parameters
```
shift = [-1, -3, -1, -1, -1,  1, -2, -2, -1, -3, -1]
dir   = [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0]
z     = 1/1
dim   = 6
box   = 6
```

### Advancing root
g₃(n) = n - 1 (denominator root, position 8 in shift vector)

### Recurrence
```
a_{n+6} = -(n-1)·a_{n+3} + 3·a_{n+4} + (n-1)·a_{n+5}
```

Coefficients grow linearly with n:
- n=1:   a_{n+6} =  0·a_{n+3} + 3·a_{n+4} +  0·a_{n+5}
- n=5:   a_{n+6} = -4·a_{n+3} + 3·a_{n+4} +  4·a_{n+5}
- n=10:  a_{n+6} = -9·a_{n+3} + 3·a_{n+4} +  9·a_{n+5}
- n=50:  a_{n+6} = -49·a_{n+3} + 3·a_{n+4} + 49·a_{n+5}
- n=100: a_{n+6} = -99·a_{n+3} + 3·a_{n+4} + 99·a_{n+5}

### Roots at n=1
f = [0, -2, 0, 0, 0, 2],  g = [0, 0, 2, -1, 1]

### Roots at n=10
f = [0, -2, 0, 0, 0, 2],  g = [0, 0, 11, -1, 1]

### Convergence verification
| N | bits of accuracy | |L - T| |
|---|---|---|
| 100 | 1,248 | 1.96e-376 |
| 200 | 2,889 | 2.38e-870 |
| 500 | 8,532 | 3.76e-2569 |
| 1000 | 19,056 | 4.55e-5737 |
| 2000 | 42,102 | 8.24e-12675 |

Bits scale linearly with N → polynomial convergence (δ ≈ 0).

### Discovery
Found by RL-guided evolutionary search (rl_e_hunt.py) at generation 266,
~9 minutes into a 1-hour run. The search used bits-of-accuracy reward
against 72 e-related targets, with diversity pressure and two-depth
convergence verification to filter false positives.

---

## Common structure

Both hits share:
- Same CMF family (6F5 companion, dim=6, z=1/1)
- Single advancing denominator root
- Polynomial convergence (δ_blind ≈ 0, bits ∝ N)
- Limit = e - p/q for rational p/q
- Recurrence coefficients polynomial in n (degree 1)
- PSLQ relation: a·L + b + c·e = 0 (linear in e)

The e-2 hit has shift[5]=1 (f₅ root = n+2), the e-3/2 hit has shift[1]=-3
and shift[9]=-3 (different root structure, g₃ advancing).

## Also discovered (not e-related)

### Golden ratio: L = φ = (1+√5)/2
```
shift = [-1, -1, 0, -1, -1, 1, -2, -2, -1, -2, 1]
dir   = [ 0,  0,  0,  0,  0, 1,  0,  0,  0,  0, 1]
z     = 1/1
```
Two advancing roots (f₅ and g₄) produce constant Fibonacci recurrence:
a_{n+6} = a_{n+4} + a_{n+5}
δ = 1.0 (Roth bound for quadratic irrationals — optimal)
PSLQ: [2, -1, -1] on [L, 1, √5]
