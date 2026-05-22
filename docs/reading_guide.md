# Reading guide — from the BCSM papers to `BCSModels.jl`

This page is for readers who want to **read the original BCSM papers with
the code open in another window**, and have the two agree line by line.
It is not a replacement for [`docs/theory.md`](theory.md) (the
self-contained derivation) — it is a cross-reference.

If you have ten minutes, read §1, §3, and §6.

---

## 1. Suggested reading order

| # | Source | Why | Approx. effort |
|---|---|---|---|
| 1 | **Fox (2024)** *JEBS* §1–3 | The cleanest entry point: single-layer IRT-BCSM, small-sample motivation. | 1 h |
| 2 | **`docs/theory.md` §1–6** | Same model, same derivation, but with every algebraic step shown and our notation. | 1–2 h |
| 3 | **`src/models/irt_bcsm.jl`** + **`src/mcmc/gibbs_irt.jl`** | Read alongside theory §6. Every block of code maps to one subsection (see §4 below). | 30 min |
| 4 | **Fox, Wenzel & Klotzke (2021)** *JEBS* | The disjoint-testlet generalisation; same machinery, multiple layers. | 1 h |
| 5 | **`src/mcmc/gibbs_testlet.jl`** | Same shape as the IRT sampler with per-layer ``Q_t`` and per-layer ``\psi_t = 1/k_t``. | 15 min |
| 6 | **Klotzke & Fox (2019a)** *Frontiers in Psychology* §3–4, esp. **Eq. 11** | The general (non-disjoint, non-binary ``\mathbf{u}_t``) BCSM. Our Remark 6.2 shows why the disjoint case is the simplification we implement. | 1 h |
| 7 | **`docs/theory.md` §7** + **`src/vb/cavi_irt.jl`** | Our original contribution: mean-field CAVI. Not in any of the papers — read after the Gibbs is solid. | 1–2 h |

---

## 2. Notation translation

The papers use slightly different conventions. The table below is the
"Rosetta stone." Code symbols are the names you will literally find in
`src/`.

| Concept | This package / `theory.md` | Fox (2024) | Fox, Wenzel & Klotzke (2021) | Klotzke & Fox (2019a) | Code symbol |
|---|---|---|---|---|---|
| Latent utility | ``Z_{ij}`` | ``Z_{ij}`` | ``Z_{ij}`` | ``Z_{ij}`` | `Z[i,j]` |
| Item difficulty | ``b_j`` (sign: ``Z = -b + \varepsilon``) | ``b_j`` | ``b_j`` | ``b_j`` | `b[j]` |
| Covariance component | ``\theta_t`` | ``\theta`` (single) | ``\theta_t`` | ``\theta_t`` | `θ[t]` |
| Layer indicator | ``\mathbf{u}_t \in \{0,1\}^K`` (binary) | ``\mathbf{1}_K`` (implicit) | binary block | general real | `U[:, t]` |
| Testlet size | ``k_t = \mathbf{u}_t^{\!\top}\mathbf{u}_t`` | ``K`` | ``k_t`` | — | `sum(U[:, t])` |
| TSIG shift | ``\psi_t = 1/k_t`` | (often implicit) | implicit | ``\psi_t^{(\mathrm{iter})} = 1/c_t``, Eq. 11 | `model.ψ` |
| TSIG truncation | ``\mathrm{tr}_t = -1/k_t`` | ``-1/K`` | ``-1/k_t`` | layer-dependent | `tr1` / `tr_t` |
| Sherman–Morrison constant | ``c_t = \mathbf{u}_t^{\!\top}\boldsymbol{\Sigma}_{-t}^{-1}\mathbf{u}_t`` | ``K`` (constant) | ``k_t`` (constant) | ``c_t`` (iter-varying) | inside `inv_covariance` |
| Per-layer sufficient stat | ``Q_t = \sum_i (\mathbf{u}_t^{\!\top}\mathbf{R}_i)^2`` | ``Q`` (single layer) | ``Q_t`` | ``Q_t`` | local `Q` |

The single biggest source of confusion when reading two papers side by
side is that Fox (2024) writes ``K`` everywhere because there is only one
testlet (the whole test). In our code that ``K`` is `model.K` *and*
``k_t = K`` *and* the Sherman–Morrison constant — all the same number.
The moment you move to Testlet-BCSM you have to distinguish them.

---

## 3. The single most important implementation note

> **``\psi_t = 1/k_t`` is the unique shift under which the conditional posterior on
> ``\theta_t`` is a closed-form inverse-gamma.** Any other value (including
> the natural-looking ``2/K`` that some readers reach for first) makes the
> posterior *not* IG, and breaks the Gibbs draw.

The math: see `docs/theory.md` Proposition 5.1, Remark 6.1, and §6.3.
The code constraint:

- [`src/models/irt_bcsm.jl`](../src/models/irt_bcsm.jl) line 29 — default `ψ = 1/K`.
- [`src/models/testlet_bcsm.jl`](../src/models/testlet_bcsm.jl) — default `ψ_scale = 1.0` so ``\psi_t = 1/k_t``.

If you change these, the sampler will still run, but the chain will
collapse onto the truncation boundary and `mean(samples_θ)` will not
recover the truth. This is the only "gotcha" in the package; everything
else follows the papers verbatim.

This is also the easiest place to misread the literature: Klotzke & Fox
(2019a) Eq. 11 states the *general* form ``\psi_t^{(\mathrm{iter})} = 1/c_t``
with ``c_t = \mathbf{u}_t^{\!\top}\boldsymbol{\Sigma}_{-t}^{-1}\mathbf{u}_t``, which
*does* reduce to ``1/k_t`` for the disjoint binary case but is easy to
miss if you skip from Fox (2024)'s implicit single-layer setup directly
to multi-layer Testlet-BCSM.

---

## 4. Equation-to-code map

### 4.1 Fox (2024) IRT-BCSM (single layer)

| Paper equation | Concept | `theory.md` | Code |
|---|---|---|---|
| Eq. 1 (covariance) | ``\boldsymbol{\Sigma} = \mathbf{I} + \theta\,\mathbf{1}\mathbf{1}^{\!\top}`` | §3 (1.1) | [`additive_covariance.jl`](../src/covariance/additive_covariance.jl) `AdditiveCovariance` |
| Eq. 2 (probit link) | ``Y_{ij} = \mathbf{1}\{Z_{ij} > 0\}`` | §4 | implicit in `sample_latent_z!` |
| Eq. 3 (Albert–Chib) | ``Z_{ij} \mid Y_{ij}=1 \sim \mathcal{TN}(\mu_{ij},\sigma^2_{ij},0,\infty)`` | §6.1 (6.1) | [`latent_utility.jl`](../src/mcmc/latent_utility.jl) `sample_latent_z!` |
| Eq. 4 (``\mathbf{b}`` update) | ``\mathbf{b} \mid \cdot \sim \mathcal{N}_K(\mathbf{m}_b, \mathbf{V}_b)`` | §6.2 (6.3) | [`gibbs_irt.jl`](../src/mcmc/gibbs_irt.jl) lines 91–99 |
| Eq. 5 (TSIG prior) | ``\theta + \psi \sim \mathrm{InvGamma}(\alpha_0,\beta_0)\,\mathbf{1}\{\theta>-1/K\}`` | §5.2 (5.2) | [`truncated_shifted_inverse_gamma.jl`](../src/distributions/truncated_shifted_inverse_gamma.jl) |
| Eq. 6 (``\theta`` conditional posterior) | ``x = \theta + 1/K \sim \mathrm{InvGamma}(\alpha_0+N/2,\,\beta_0+Q/(2K^2))`` | §5.4 (5.5) / §6.3 (6.8) | [`gibbs_irt.jl`](../src/mcmc/gibbs_irt.jl) lines 101–114 |

The single Gibbs sweep in `gibbs_irt.jl` lines 86–129 is a direct line-by-line
translation of Eqs. 3–6.

### 4.2 Fox, Wenzel & Klotzke (2021) Testlet-BCSM

The testlet paper generalises the above to ``T`` disjoint binary layers
with ``\boldsymbol{\Sigma} = \mathbf{I} + \sum_t \theta_t \mathbf{u}_t \mathbf{u}_t^{\!\top}``.
The key fact (and the reason the Gibbs is exactly as cheap as the single-layer
case) is:

> **For disjoint binary ``\mathbf{u}_t``, each ``\mathbf{u}_t`` is an eigenvector
> of ``\boldsymbol{\Sigma}_{-t}`` with eigenvalue 1**, so
> ``c_t = \mathbf{u}_t^{\!\top}\boldsymbol{\Sigma}_{-t}^{-1}\mathbf{u}_t = k_t``
> *regardless of the other ``\boldsymbol{\theta}_{-t}``*.

(`docs/theory.md` §6.3 derives this; Remark 6.2 shows what breaks if
``\mathbf{u}_t`` is non-disjoint.) Consequently:

| Paper concept | Code |
|---|---|
| Per-layer ``\psi_t = 1/k_t`` | [`testlet_bcsm.jl`](../src/models/testlet_bcsm.jl) `shift_vector` |
| Per-layer truncation ``-1/k_t`` | [`testlet_bcsm.jl`](../src/models/testlet_bcsm.jl), used in [`gibbs_testlet.jl`](../src/mcmc/gibbs_testlet.jl) loop |
| Per-layer ``Q_t`` | [`gibbs_testlet.jl`](../src/mcmc/gibbs_testlet.jl) `Q[t]` accumulator |
| Per-layer IG draw | [`gibbs_testlet.jl`](../src/mcmc/gibbs_testlet.jl), one `rand_tsig` per `t` |

### 4.3 Klotzke & Fox (2019a) general BCSM

The 2019a paper is the most general (non-disjoint, non-binary ``\mathbf{u}_t``).
We implement only the disjoint-binary specialisation, but the general
derivation in Eq. 11 there is what justifies the "shift = 1/c_t" choice
in the first place.

- 2019a Eq. 11 ↔ `theory.md` Remark 6.2.
- We do not implement the iter-varying ``\psi_t^{(\mathrm{iter})} = 1/c_t``
  fallback; if you need non-disjoint or non-binary layers, please open an
  issue (it is a small extension but currently untested).

---

## 5. CAVI (no paper reference — original contribution)

The variational engine is novel for this package. There is no equation
in any of the BCSM papers to cross-reference. The derivation is entirely
in `docs/theory.md` §7, and the code mirrors that derivation:

| Algorithmic step | `theory.md` | Code |
|---|---|---|
| Mean-field factorisation (7.1) | §7.1 | [`cavi_irt.jl`](../src/vb/cavi_irt.jl) struct fields |
| ``q(Z_{ij})`` update — truncated normal moments | §7.2 | [`truncated_normal_moments.jl`](../src/distributions/truncated_normal_moments.jl) `truncnorm_moments` |
| ``q(b_j)`` update — Gaussian | §7.3 | [`cavi_irt.jl`](../src/vb/cavi_irt.jl) `update_qb!` |
| ``q(\theta_t)`` update — TSIG with plug-in ``\mathbb{E}_q[\boldsymbol{\Lambda}] \approx \boldsymbol{\Lambda}(\mathbb{E}_q[\boldsymbol{\theta}])`` | §7.4 + §7.6 | [`cavi_irt.jl`](../src/vb/cavi_irt.jl) `update_qθ!` |
| ELBO assembly | §7.5 | [`cavi_irt.jl`](../src/vb/cavi_irt.jl) `elbo` |
| Structural bias explanation (why CAVI under-estimates ``\theta_t``) | §7.7 | observed in [`docs/param_recovery.md`](param_recovery.md) |

If you are coming from another VB-IRT paper (e.g. Cho et al. 2021 on
Gaussian variational MIRT, or Polson–Scott–Windle 2013 on
Pólya–Gamma logistic VB), see `theory.md` Appendix B for the
side-by-side.

---

## 6. End-to-end trace: one Gibbs sweep, line by line

For the reader who wants the tightest possible mapping between Fox
(2024) Eqs. 3–6 and the code, here is one iteration of the IRT-BCSM
Gibbs sampler annotated with both:

```julia
# --- src/mcmc/gibbs_irt.jl line 86 onward ---------------------------------
for it in 1:total_iters
    # Fox (2024) Eq. 3 — Albert–Chib + Geweke sweep over the latent utilities
    sample_latent_z!(rng, Z, Y, b, Λ)                                  # §6.1

    # Fox (2024) Eq. 4 — Gaussian conditional on b
    Vinv = Symmetric(N .* Λ .+ Diagonal(fill(1 / model.τ²_b, K)))      # §6.2
    V    = inv(Vinv)
    ssum = vec(sum(Z, dims=1))
    rhs  = (model.μ_b / model.τ²_b) .* ones(K) .- Λ * ssum
    m_b  = V * rhs
    Cb   = cholesky(Symmetric((V + V') / 2))
    b   .= m_b .+ Cb.U' * randn(rng, K)

    # Fox (2024) Eq. 6 — TSIG conditional on θ
    # Q = Σ_i (1ᵀ r_i)² with r_i = z_i + b
    Q = 0.0
    @inbounds for i in 1:N, j in 1:K
        # ...inner sum, see source for the exact loop
    end
    d  = TruncatedShiftedInverseGamma(α₀ + N/2,
                                      β₀ + Q / (2 * K^2),               # §5.4 (5.5)
                                      ψ,                                # = 1/K — §3
                                      tr1)                              # = -1/K
    θ[1] = rand_tsig(rng, d)                                            # §6.3

    # Refresh Σ⁻¹ via Sherman–Morrison
    Λ = inv_covariance(AdditiveCovariance(σ², θ, U))                    # Woodbury, §6.4
end
```

The whole Testlet-BCSM sampler is the same code with a per-`t` loop
around the ``Q``/`rand_tsig` block, using `model.ψ[t]` and `-1/k_t` as
the truncation.

---

## 7. Where to look for each section of `theory.md`

For readers who already know the theory and just want the code:

| `theory.md` section | What it derives | Implementation file |
|---|---|---|
| §3 (model setup) | ``\boldsymbol{\Sigma}`` decomposition | `src/covariance/additive_covariance.jl` |
| §5.2 (TSIG prior) | TSIG density, moments, sampling | `src/distributions/truncated_shifted_inverse_gamma.jl` |
| §5.4 + §6.3 (canonical shift) | Why ``\psi_t = 1/k_t`` | `src/models/*_bcsm.jl` defaults |
| §6.1 (Albert–Chib + Geweke) | Univariate truncated normal sweep | `src/mcmc/latent_utility.jl` |
| §6.2 (``\mathbf{b}`` update) | Gaussian conditional | `gibbs_irt.jl` lines 91–99 |
| §6.3 (``\theta`` update) | TSIG conditional with closed-form ``Q_t`` | `gibbs_irt.jl` lines 101–114 |
| §6.4 (Sherman–Morrison refresh) | ``\boldsymbol{\Lambda}`` from ``\boldsymbol{\Sigma}`` | `src/covariance/additive_covariance.jl` `inv_covariance` |
| §7.1–§7.6 (CAVI) | Mean-field updates and ELBO | `src/vb/cavi_irt.jl`, `src/vb/cavi_testlet.jl` |
| §7.7 (structural bias of mean-field) | Predicts the empirical bias | observed in `param_recovery.md` |

---

## 8. Where the code does *not* follow the papers

We try hard to keep the code aligned with the literature. The known
exceptions are:

1. **`\psi_t = 1/k_t` chosen explicitly.** Fox (2024) is implicit; we
   document and enforce it (see §3 above and `theory.md` Remark 6.1).
2. **Disjoint binary ``\mathbf{u}_t`` only.** Klotzke & Fox (2019a) Eq. 11
   allows general ``\mathbf{u}_t`` with iteration-varying shift; we do
   not implement that path. See `theory.md` Remark 6.2.
3. **Unconstrained residual variances disabled by default.** The
   identifiability fix ``\sigma_j^2 \equiv 1`` is the default; the
   Klotzke & Fox (2019a) Eq. 12 TSIG update on ``\sigma_j^2`` is in the
   distributions module but not wired into the Gibbs sweep.
4. **Mean-field CAVI is novel.** No paper to cross-check. Validated
   empirically against the Gibbs sampler in `mcmc_vs_vb.md` and
   `param_recovery.md`.

---

## 9. Quick lookup — "I'm reading paper X, where is Y in the code?"

- "What does the Σ-update look like?" → `inv_covariance` in `additive_covariance.jl`.
- "Where is the truncated normal sweep?" → `sample_latent_z!` in `latent_utility.jl`.
- "Where is the ``\theta_t`` conditional draw?" → search `rand_tsig(` in `gibbs_*.jl`.
- "Where is the TSIG density?" → `logpdf_tsig` in `truncated_shifted_inverse_gamma.jl`.
- "Where do you compute the ELBO?" → `elbo(...)` in `cavi_irt.jl` / `cavi_testlet.jl`.
- "Where is the simulator that I can use to reproduce the paper?" →
  `simulate_irt_bcsm` / `simulate_testlet_bcsm` in `src/data/simulation.jl`.
