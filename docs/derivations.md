# BCSM derivations for Gibbs and CAVI

This note collects the algebra used by the implementations in `src/mcmc/` and
`src/vb/`. Notation follows Klotzke & Fox (2019a, *Frontiers in Psychology*,
Eqs. 1, 6, 7, 10, 11, 12) and Fox (2024, *JEBS*, Eqs. 1–6) with minor changes
to remain consistent across model variants.

## 1. Model

### 1.1 Likelihood (Albert–Chib probit augmentation)

For respondents `i = 1, …, N` and items `j = 1, …, K`, let `y_{ij} ∈ {0,1}` be the
observed response. Introduce latent utilities `z_{ij}` with

    z_{i} = -b + ε_{i},          ε_{i} ~ N_K(0, Σ),
    y_{ij} = 1[z_{ij} > 0],

so that the marginal probit likelihood is `Φ(b_j √Σ_jj^{-1})`-shaped after
marginalising over the layer-specific latents. Here `b ∈ ℝ^K` collects item
intercepts (item difficulties with a sign convention).

### 1.2 Additive covariance

The covariance `Σ ∈ ℝ^{K×K}` is additive in `T` rank-1 structured layers,

    Σ = Σ₀ + Σ_{t=1}^{T} θ_t u_t u_tᵀ,                            (1)

where `Σ₀ = diag(σ₁², …, σ_K²)` is the item-specific residual variance and
`u_t ∈ {0,1}^K` is a binary classification vector (testlet membership, group,
discrimination support). For probit identifiability we fix `σ_j² = 1` for all `j`
in the basic IRT-BCSM and Testlet-BCSM. Component `θ_t ∈ ℝ` can be negative.

### 1.3 Special cases used in this package

- **IRT-BCSM (Fox 2024).** A single layer with `u₁ = 1_K` reproduces the marginal
  covariance of a 1PL probit model after integrating out the person factor; with
  `u₁ = a` (binary loading pattern) it reproduces a 2PL-like marginal structure.
  In our small-sample implementation we set `u₁ = 1_K` and treat item-specific
  discrimination through additional layers when needed.

- **Testlet-BCSM (Fox, Wenzel & Klotzke 2021).** One layer per testlet `t ∈ T_g`
  with `u_t = 1[item j ∈ testlet t]`. Layers commute and update independently
  conditional on the latent utilities.

## 2. Positive-definiteness and the truncation bound

Adding a rank-1 layer `θ_t u_t u_tᵀ` to a positive-definite `Σ_{t-1}` keeps `Σ_t`
positive definite iff

    θ_t > tr_t,   tr_t = - 1 / (u_tᵀ Σ_{t-1}^{-1} u_t).            (2)

The bound `tr_t` is negative; positive `θ_t` is always admissible.

Two cheap formulae are useful:

- If `Σ_{t-1}` is itself in BCSM form, the Sherman–Morrison identity gives
  `Σ_{t-1}^{-1}` updates without forming `Σ_{t-1}`.

- For the IRT-BCSM with `Σ₀ = I`, the bound is `tr_1 = -1/K`.

## 3. Priors

Each `θ_t` receives a **truncated shifted inverse-gamma** prior

    p(θ_t) ∝ (θ_t + ψ_t)^{-(α₀+1)} exp(-β₀ / (θ_t + ψ_t)) · 1[θ_t > tr_t]. (3)

The shift `ψ_t ≥ -tr_t > 0` makes the prior proper on `(tr_t, ∞)` and induces a
proper marginal posterior. **The closed-form conditional posterior on
`x = θ_t + ψ_t` collapses to a plain inverse-gamma only when `ψ_t = -tr_t`
exactly** — for binary `u_t` and `Σ₀ = I`, this is `ψ_t = 1/k_t` (k_t the
testlet size, or K in the single-layer IRT-BCSM). With any other ψ the
conditional retains a residual log-correction term and the closed-form
Gibbs/CAVI updates below do *not* apply. The default in the implementation is
`ψ_t = 1/k_t` (Klotzke & Fox 2019a Eq. 7 minimal-slack choice).

Default hyperparameters in the implementation: `α₀ = β₀ = 1`.

Item intercepts: `b_j ~ N(μ_b, τ_b²)`, default `μ_b = 0, τ_b² = 4` (weakly
informative on the probit scale, ±2σ ≈ ±4 covers most of the difficulty range).

## 4. Closed-form conditional posteriors (Gibbs)

### 4.1 Latent utilities `z_{ij}`

Conditional on item parameters and Σ, the joint distribution of `z_i` is
truncated multivariate normal,

    z_i | y_i, b, Σ ~ TN_K(-b, Σ; constraints from y_i).

We sample with a one-at-a-time Geweke (1991) Gibbs scan inside each iteration,
using the Sherman–Morrison form of `Σ^{-1}` so no `K × K` inverse is materialised.
For diagonal `Σ₀ = I` and rank-1 layers, the conditional means and variances are
each O(K) per coordinate.

### 4.2 Item intercepts `b`

With the working data `r_{ij} = -z_{ij}` (so that `r_i ~ N(b, Σ)`),

    b | rest ~ N_K(m, V),
    V^{-1} = N · Σ^{-1} + τ_b^{-2} I,
    m = V (Σ^{-1} Σ_i r_i + τ_b^{-2} μ_b 1).

For Σ in BCSM form we evaluate `Σ^{-1}` via repeated Sherman–Morrison.

### 4.3 Covariance components `θ_t`

After centring `z̃_i = z_i - (-b) = z_i + b`, the layer-`t` conditional is also a
truncated shifted inverse-gamma. Define

    SSB_t = Σ_i (u_tᵀ z̃_i)² / ‖u_t‖²,  (sum of squares aligned with `u_t`),

then

    θ_t | rest ~ tsIG⁺( α₀ + N/2,  β₀ + SSB_t/2,  ψ_t,  tr_t ).    (4)

(Eq. (4) is Klotzke & Fox 2019a, Eq. 11, specialised to `Σ₀ = I` and binary `u_t`.)

The truncation bound `tr_t` is updated each iteration from the current other
layers.

### 4.4 Residual variances `σ_j²` (when not fixed)

If we relax `σ_j² = 1`,

    σ_j² | rest ~ tsIG⁺( α₀ + N/2,  β₀ + SSW_j/2,  Σ_t θ_t u_{t,j},  0 ), (5)

with `SSW_j = Σ_i (z̃_{ij} - Σ_t θ_t u_{t,j} ē_t)²` and `ē_t` the layer averages.
For probit identifiability we keep `σ_j² = 1` in the released models and noting
that Eq. (5) is implemented but disabled by default.

## 5. Mean-field CAVI derivation

The mean-field family factorises as

    q(z, b, θ) = ∏_{i,j} q(z_{ij}) · ∏_j q(b_j) · ∏_t q(θ_t).

Setting `log q*(·) = E_{-·}[log p(y, z, b, θ)] + const` and exploiting the
conditional conjugacy above gives:

### 5.1 `q(z_{ij})` — truncated normal

    q(z_{ij}) = TN( E_q[m_{ij}] , E_q[v_{ij}] ; sign(y_{ij}-1/2) ),

where the conditional mean and variance are obtained from `Σ` partitioning. For
a single rank-1 layer with `u = 1_K`,

    m_{ij} = -E_q[b_j] + (E_q[θ] / (1 + E_q[θ]·(K-1)/K))
             · (Σ_{k≠j} (E_q[z_{ik}] + E_q[b_k]) - (K-1)·(-E_q[b_j])),

and an analogous variance. We keep the first two moments `E_q[z_{ij}]` and
`E_q[z_{ij}²]` since only those enter downstream updates.

The moments of a univariate truncated normal `TN(m, v; sign)` are standard:

    α = -m / √v,
    if sign = +1: λ = φ(α) / (1 - Φ(α));    μ = m + √v λ;  ν = v(1 - λ(λ - α))
    if sign = -1: λ = -φ(α) / Φ(α);         μ = m + √v λ;  ν = v(1 - λ(λ - α))

### 5.2 `q(b_j)` — Gaussian

`q(b_j) = N(m_{b,j}, v_{b,j})` with

    v_{b,j}^{-1} = N · E_q[(Σ^{-1})_{jj}] + τ_b^{-2},
    m_{b,j}      = v_{b,j} · ( - E_q[(Σ^{-1} r̄)_j] · N + τ_b^{-2} μ_b ),

where `r̄_j = (1/N) Σ_i (- E_q[z_{ij}])`. We approximate
`E_q[(Σ^{-1})_{jj}] ≈ (Σ̂^{-1})_{jj}` evaluated at `θ̂ = E_q[θ]`. The error of this
first-order plug-in is O(Var_q[θ]/Σ̂²) and is empirically small for moderate
sample sizes (Wand 2017).

### 5.3 `q(θ_t)` — truncated shifted inverse-gamma

The form (4) survives the variational update; the only change is replacing
sufficient statistics by their `q`-expectations,

    α̃_t = α₀ + N/2,
    β̃_t = β₀ + (1/2) Σ_i E_q[(u_tᵀ z̃_i)²] / ‖u_t‖²,
    q(θ_t) = tsIG⁺(α̃_t, β̃_t, ψ_t, tr_t(θ̂_{-t})).

Required moments `E_q[θ_t]` and `E_q[θ_t²]` are computed via incomplete-gamma
integrals against the truncation `tr_t`.

## 6. CAVI sweep and ELBO

A single CAVI iteration sweeps (5.1) → (5.2) → (5.3) and updates the truncation
bound from the current `E_q[θ_{-t}]`. Convergence is monitored by the ELBO

    ELBO(q) = E_q[log p(y, z, b, θ)] - E_q[log q],

evaluated in closed form using the moment expressions above (truncated normal
log-normaliser, Gaussian entropy, truncated shifted inverse-gamma normaliser).

## 7. Implementation notes

- All `Σ` operations are kept in O(N K T) per iteration by exploiting binary
  `u_t` and Sherman–Morrison updates.
- Geweke univariate truncated-MVN sampling is sufficient because the conditional
  variances factorise (no full Cholesky needed).
- Numerical guards: clamp `tr_t` away from -∞ with `tr_t ← max(tr_t, -10/K)`
  to keep the prior support bounded for very small `K`.

## 8. Identifiability

We fix `σ_j² = 1` (probit scale) and place a proper Gaussian prior on `b`. The
covariance scale of `Σ` is then identified through the truncation `tr_t < 0`
together with the proper shifted prior. No further rotation is needed in the
IRT-BCSM/Testlet-BCSM specifications.
