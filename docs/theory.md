# Bayesian Covariance Structure Models for Item Response Theory: Gibbs and Mean-field Variational Inference

**Implementation reference for the [BCSModels.jl](../README.md) Julia package.**

> **Status note.** This document collects the algebra that the source code
> implements line-by-line. A peer-reviewed treatment of the same material
> is in preparation; the canonical citable manuscript will be the arXiv
> preprint and the eventual journal version (see
> [`CITATION.cff`](../CITATION.cff) and the README "Citation" section for
> the up-to-date pointer). Until then, please cite the software itself if
> you use these derivations directly.

---

## Abstract

We present the Bayesian Covariance Structure Model (BCSM) of Fox, Mulder,
Klotzke and colleagues as it applies to item response data, together with two
posterior inference engines that we implement and study: the closed-form
Gibbs sampler used throughout the BCSM literature and a new mean-field
coordinate-ascent variational Bayes (CAVI) algorithm. We carry the derivations
in full — including the algebraic identity that makes the conditional
posterior of every covariance component a *pure inverse-gamma* under one
specific choice of the prior shift, the Albert–Chib augmentation that makes
both samplers tractable, and the propagation of the plug-in approximation
through the ELBO. The CAVI derivation is, to our knowledge, the first for the
BCSM family. We close with a discussion of identifiability, the
positive-definiteness constraint via Sherman–Morrison, and the empirically
observed bias of mean-field VB on multi-layer BCSM.

---

## 1. Introduction

Multilevel and testlet item-response-theory (IRT) models induce a structured
*marginal* covariance on the latent response utilities once their random
effects are integrated out. The Bayesian Covariance Structure Model (BCSM;
Klotzke & Fox, 2019a, b; Fox, Wenzel & Klotzke, 2021; Fox, 2024) takes this
marginal as the primary object of inference: rather than estimate
random-effect variances $\sigma_b^2 \geq 0$ in a hierarchical formulation,
BCSM places a proper prior directly on the covariance parameters $\theta_t$
of the additive decomposition
$$
\boldsymbol{\Sigma} \;=\; \boldsymbol{\Sigma}_0 \;+\; \sum_{t=1}^{T} \theta_t \, \mathbf{u}_t \mathbf{u}_t^{\!\top},
\qquad \theta_t \in \mathbb{R}.
\tag{1.1}
$$
The benefits are well documented: the parameters $\theta_t$ may be *negative*
(unattainable in a random-effects formulation), the conditional posteriors
are closed-form inverse-gammas, and Bayes factors for covariance-structure
hypotheses are immediate.

The standard inference engine for BCSM is a Gibbs sampler. We re-derive it
below. The novelty of this document is a fully closed-form mean-field
variational counterpart, derived under the same conditional conjugacy that
makes the Gibbs sampler tractable. Both engines are implemented in the
companion package `BCSModels.jl`.

We focus on the dichotomous-response IRT setting because (i) it is the
target application driving the design of `BCSModels.jl`, and (ii) it admits the
Albert–Chib augmentation that lets us treat probit IRT inside a Gaussian-MVN
framework. The polytomous case is a straightforward extension and is
sketched in Section 9.

---

## 2. Notation and preliminaries

| Symbol | Meaning |
|---|---|
| $N$ | number of respondents |
| $K$ | number of items |
| $T$ | number of additive covariance layers |
| $\mathbf{Y} \in \{0,1\}^{N \times K}$ | observed binary responses, $Y_{ij}$ for respondent $i$ on item $j$ |
| $\mathbf{Z} \in \mathbb{R}^{N \times K}$ | latent utilities (Albert–Chib augmentation) |
| $\mathbf{b} \in \mathbb{R}^K$ | item difficulties (with a sign convention; see below) |
| $\mathbf{u}_t \in \mathbb{R}^K$ | indicator vector for layer $t$ |
| $\theta_t \in \mathbb{R}$ | covariance component for layer $t$ |
| $\boldsymbol{\Sigma} \in \mathbb{R}^{K \times K}$ | marginal covariance, equation (1.1) |
| $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}$ | precision matrix |
| $\boldsymbol{\Sigma}_0 = \mathrm{diag}(\sigma_1^2, \dots, \sigma_K^2)$ | base diagonal covariance |
| $k_t = \mathbf{u}_t^{\!\top} \mathbf{u}_t$ | layer size (number of 1's when $\mathbf{u}_t$ is binary) |
| $c_t = \mathbf{u}_t^{\!\top} \boldsymbol{\Sigma}_{-t}^{-1} \mathbf{u}_t$ | layer-$t$ "Sherman–Morrison" scalar; see Lemma 4.1 |
| $\mathrm{tr}_t = -1/c_t$ | positive-definiteness lower bound for $\theta_t$ |

We assume **disjoint binary** $\mathbf{u}_t$'s: each item $j$ belongs to at
most one layer $t$. This covers the IRT-BCSM (single layer $\mathbf{u}_1 =
\mathbf{1}_K$) and the Testlet-BCSM (one layer per testlet). Section 9.3
sketches the extension to overlapping layers.

We will repeatedly use the **matrix determinant lemma** and **Sherman–Morrison
formula**: for $\mathbf{A} \succ 0$ and $\mathbf{u} \in \mathbb{R}^K$,
$$
\det(\mathbf{A} + \theta \, \mathbf{u}\mathbf{u}^{\!\top})
  \;=\; \det(\mathbf{A}) \, \bigl(1 + \theta \, \mathbf{u}^{\!\top} \mathbf{A}^{-1} \mathbf{u}\bigr),
\tag{2.1}
$$
$$
(\mathbf{A} + \theta \, \mathbf{u}\mathbf{u}^{\!\top})^{-1}
  \;=\; \mathbf{A}^{-1} \;-\; \frac{\theta}{1 + \theta \, \mathbf{u}^{\!\top} \mathbf{A}^{-1} \mathbf{u}} \, \mathbf{A}^{-1}\mathbf{u}\mathbf{u}^{\!\top}\mathbf{A}^{-1}.
\tag{2.2}
$$

---

## 3. Model specification

### 3.1 Probit IRT with Albert–Chib augmentation

We adopt the standard latent-utility representation. Let
$$
Z_{ij} \;=\; -b_j + \varepsilon_{ij}, \qquad \boldsymbol{\varepsilon}_i \sim \mathcal{N}_K(\mathbf{0}, \boldsymbol{\Sigma}),
\qquad Y_{ij} \;=\; \mathbf{1}\{Z_{ij} > 0\}.
\tag{3.1}
$$
The sign convention puts the difficulty $b_j$ inside the *mean* with a
negative sign, so that
$$
\Pr(Y_{ij} = 1 \mid b_j, \sigma_j^2)
  \;=\; \Phi\!\left(\frac{-b_j}{\sqrt{\sigma_j^2 + \mathbf{u}^{(j)}\boldsymbol{\theta}}}\right),
$$
where $\mathbf{u}^{(j)}\boldsymbol{\theta} = \sum_t \theta_t \, u_{t,j}^2$ is
the contribution of all layers covering item $j$. With $\sigma_j^2 \equiv 1$
(see Section 8 on identifiability), large $b_j$ ⇒ low probability of correct.

Equivalently, in matrix form,
$$
\mathbf{Z}_i \;\sim\; \mathcal{N}_K(-\mathbf{b}, \boldsymbol{\Sigma}), \qquad i = 1, \dots, N. \tag{3.2}
$$

### 3.2 BCSM as marginalised multilevel IRT

To motivate (1.1), consider the multilevel 2PL probit with a single random
person effect:
$$
Z_{ij} \;=\; -b_j + a_j \, \eta_i + \xi_{ij},
\qquad \eta_i \sim \mathcal{N}(0, \tau), \quad \xi_{ij} \sim \mathcal{N}(0, 1),
$$
with $\eta_i \perp \xi_{ij}$. Integrating out $\eta_i$ gives a marginal
covariance
$$
\mathrm{Cov}(Z_{ij}, Z_{ik}) \;=\; \delta_{jk} + a_j a_k \tau,
$$
i.e. $\boldsymbol{\Sigma} = \mathbf{I}_K + \tau \, \mathbf{a}\mathbf{a}^{\!\top}$.
This is exactly the single-layer BCSM with $\mathbf{u}_1 = \mathbf{a}$ and
$\theta_1 = \tau$. Crucially:

> In the multilevel formulation $\tau \geq 0$ is enforced ($\tau$ is a
> variance). In BCSM we drop that constraint and only require
> $\boldsymbol{\Sigma}$ to be positive definite — which admits $\theta_1$
> as low as $-1/(\mathbf{a}^{\!\top}\mathbf{a}) < 0$. This is what
> Fox & Smink (2021) describe as "negative variance components" becoming
> first-class objects.

For testlets, replace the single $\eta_i$ by per-testlet random effects
$\eta_{it} \sim \mathcal{N}(0, \tau_t)$, integrate them out, and obtain
$$
\boldsymbol{\Sigma} \;=\; \mathbf{I}_K + \sum_{t=1}^T \tau_t \, \mathbf{u}_t \mathbf{u}_t^{\!\top}
$$
with $\mathbf{u}_t$ the binary testlet indicator. Setting $\theta_t = \tau_t$
recovers (1.1).

### 3.3 The BCSM marginal model used in this document

Combining (3.2) and (1.1):
$$
\boxed{\;\;
\mathbf{Z}_i \;\sim\; \mathcal{N}_K\!\bigl(-\mathbf{b}, \, \boldsymbol{\Sigma}_0 + \sum_{t=1}^T \theta_t \mathbf{u}_t \mathbf{u}_t^{\!\top}\bigr),
\quad
Y_{ij} = \mathbf{1}\{Z_{ij} > 0\}.
\;\;}
\tag{3.3}
$$
The unknowns are $(\mathbf{b}, \boldsymbol{\theta}, \boldsymbol{\sigma}^2)$
with $\boldsymbol{\theta} = (\theta_1, \dots, \theta_T)$ and
$\boldsymbol{\sigma}^2 = (\sigma_1^2, \dots, \sigma_K^2)$. The latent $\mathbf{Z}$
is the augmentation that turns posterior inference into manipulations on a
Gaussian linear model.

---

## 4. Positive-definiteness and the truncation bound

To make (3.3) a valid model we need $\boldsymbol{\Sigma} \succ 0$. Adding the
layers one at a time, this gives a sequential constraint:

**Lemma 4.1.** *Let $\boldsymbol{\Sigma}_{t-1} \succ 0$, $\mathbf{u}_t \neq
\mathbf{0}$, and $c_t := \mathbf{u}_t^{\!\top} \boldsymbol{\Sigma}_{t-1}^{-1} \mathbf{u}_t > 0$.
Then $\boldsymbol{\Sigma}_t := \boldsymbol{\Sigma}_{t-1} + \theta_t \mathbf{u}_t \mathbf{u}_t^{\!\top}
\succ 0$ if and only if*
$$
\theta_t \;>\; \mathrm{tr}_t \;:=\; -\frac{1}{c_t}.
\tag{4.1}
$$

*Proof.* By the matrix determinant lemma (2.1),
$\det(\boldsymbol{\Sigma}_t) = \det(\boldsymbol{\Sigma}_{t-1})(1 + \theta_t c_t)$.
Since $\boldsymbol{\Sigma}_{t-1} \succ 0$, $\det(\boldsymbol{\Sigma}_{t-1}) > 0$. The eigenvalues of
$\boldsymbol{\Sigma}_t$ are those of $\boldsymbol{\Sigma}_{t-1}$ except along the direction $\mathbf{u}_t$,
where the eigenvalue is shifted by $\theta_t \, c_t$. The smallest eigenvalue
is positive iff $1 + \theta_t c_t > 0$, i.e. $\theta_t > -1/c_t$. $\square$

**Remark 4.2** (*Disjoint binary $\mathbf{u}_t$*). If
$\mathbf{u}_t \in \{0,1\}^K$ and $\mathbf{u}_t^{\!\top} \mathbf{u}_s = 0$ for
all $s \neq t$, then $\mathbf{u}_t$ is an eigenvector of *every* other
layer's contribution with eigenvalue $0$, so $\boldsymbol{\Sigma}_{-t} \mathbf{u}_t = \mathbf{u}_t$
(taking $\boldsymbol{\Sigma}_0 = \mathbf{I}$). Therefore
$$
c_t \;=\; \mathbf{u}_t^{\!\top} \boldsymbol{\Sigma}_{-t}^{-1} \mathbf{u}_t \;=\; \mathbf{u}_t^{\!\top} \mathbf{u}_t \;=\; k_t,
\quad\text{so}\quad \mathrm{tr}_t \;=\; -1/k_t. \tag{4.2}
$$
The truncation bound is then *independent* of the other layers — a key
algorithmic convenience that we exploit throughout.

---

## 5. Prior specification

### 5.1 Item difficulties

We use a weakly informative Gaussian prior
$$
b_j \;\overset{\text{iid}}{\sim}\; \mathcal{N}(\mu_b, \tau_b^2),
\qquad \mu_b = 0, \quad \tau_b^2 = 4 \tag{5.1}
$$
on the probit scale (so $\pm 2 \tau_b \approx \pm 4$ covers the practical
range of item difficulty).

### 5.2 Covariance components — truncated shifted inverse-gamma

For each layer $t$ we place a *truncated shifted inverse-gamma* prior on
$\theta_t$. Defining $x = \theta_t + \psi_t$ with shift $\psi_t > -\mathrm{tr}_t$,
the prior density is
$$
p(\theta_t) \;\propto\;
   (\theta_t + \psi_t)^{-(\alpha_0 + 1)} \exp\!\left(-\frac{\beta_0}{\theta_t + \psi_t}\right)
   \, \mathbf{1}\{\theta_t > \mathrm{tr}_t\}.
\tag{5.2}
$$
Equivalently, $x \sim \mathrm{InvGamma}(\alpha_0, \beta_0)$ restricted to
$x > \mathrm{tr}_t + \psi_t > 0$.

Two properties matter for what follows:

* For *any* $\psi_t > -\mathrm{tr}_t$ this is a proper density on
  $(\mathrm{tr}_t, \infty)$, so $\theta_t$ can be negative.
* The form (5.2) is *exactly* the same family as the conditional posterior
  derived in Section 6.3 below — but only under a specific choice of
  $\psi_t$, namely (5.4) below.

### 5.3 Residual variances

We fix $\sigma_j^2 \equiv 1$ throughout this document for identifiability
(Section 8). The implementation supports the unconstrained version (a
TSIG prior on $\sigma_j^2$ with shift $\sum_t \theta_t u_{t,j}^2$ and
truncation 0; Klotzke & Fox 2019a Eq. 12) but it is disabled by default.

### 5.4 The canonical choice of the shift

The following identity is the algebraic engine of the BCSM Gibbs sampler.
It explains why a particular value of $\psi_t$ makes the conditional
posterior in $\theta_t$ collapse to a clean inverse-gamma in $x = \theta_t +
\psi_t$, with the truncation $\theta_t > \mathrm{tr}_t$ becoming exactly the
natural IG support $x > 0$. We state it as a proposition because we want
to highlight it: it is implicit in Klotzke & Fox (2019a) but, as we
discovered while implementing the sampler, the conditional is *not* in
closed form for any other choice of $\psi_t$.

**Proposition 5.1** (*Canonical shift*). *Under disjoint binary $\mathbf{u}_t$
and $\boldsymbol{\Sigma}_0 = \mathbf{I}$, set*
$$
\psi_t \;:=\; \frac{1}{k_t}. \tag{5.4}
$$
*Then with $x = \theta_t + \psi_t$, the conditional log-likelihood of
$\theta_t$ given the latent utilities reduces to*
$$
\log p(\mathbf{Z} \mid \theta_t, \cdot) \;=\; \mathrm{const} \;-\;
   \frac{N}{2} \log x \;-\; \frac{Q_t}{2 \, k_t^2 \, x},
$$
*where $Q_t := \sum_i \bigl(\mathbf{u}_t^{\!\top}(\mathbf{Z}_i + \mathbf{b})\bigr)^2$.
Consequently, combined with the prior (5.2), the conditional posterior on
$x$ is*
$$
x \;\big|\; \mathbf{Z}, \mathbf{b}, \boldsymbol{\theta}_{-t} \;\sim\;
   \mathrm{InvGamma}\!\left(\alpha_0 + \tfrac{N}{2}, \; \beta_0 + \tfrac{Q_t}{2 \, k_t^2}\right),
   \tag{5.5}
$$
*restricted to $x > 0$ (i.e. $\theta_t > \mathrm{tr}_t = -1/k_t$).*

The proof is the centrepiece of the BCSM Gibbs and is given in detail in
Section 6.3.

---

## 6. Gibbs sampler

We derive the three blocks of the standard BCSM Gibbs sweep:
$\mathbf{Z}$ (latent utilities), $\mathbf{b}$ (item difficulties), and
$\boldsymbol{\theta}$ (covariance components). The whole sweep is closed
form.

### 6.1 Latent utilities $\mathbf{Z}$ — Albert–Chib + Geweke

Given $(\mathbf{b}, \boldsymbol{\theta})$, the latent vector $\mathbf{Z}_i$
is multivariate normal with mean $-\mathbf{b}$ and covariance
$\boldsymbol{\Sigma}$, restricted to the orthant determined by the observed
$\mathbf{Y}_i$. Direct multivariate-truncated-normal sampling is expensive;
following Geweke (1991) we sample each coordinate from its univariate
conditional in a sweep.

For an MVN $\mathbf{X} \sim \mathcal{N}_K(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ with precision
$\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}$, the conditional of $X_j$ given $\mathbf{X}_{-j}$ is
Gaussian with
$$
\mathrm{Var}(X_j \mid \mathbf{X}_{-j}) \;=\; \frac{1}{\Lambda_{jj}},
\qquad
\mathbb{E}(X_j \mid \mathbf{X}_{-j}) \;=\; \mu_j - \frac{1}{\Lambda_{jj}} \sum_{k \neq j} \Lambda_{jk}(X_k - \mu_k).
\tag{6.1}
$$
Specialising to $\mathbf{X} = \mathbf{Z}_i$, $\boldsymbol{\mu} = -\mathbf{b}$:
$$
\mathbb{E}(Z_{ij} \mid \mathbf{Z}_{i,-j}) \;=\;
   -b_j \;-\; \frac{1}{\Lambda_{jj}} \sum_{k \neq j} \Lambda_{jk}\bigl(Z_{ik} + b_k\bigr).
\tag{6.2}
$$
Together with the truncation
$\mathrm{sign}(Z_{ij}) = 2 Y_{ij} - 1$, the conditional sample is from a
univariate truncated normal. (See e.g. Albert & Chib 1993 for the original
probit-regression case.)

**Algorithmic note**. We maintain the auxiliary vector
$\mathbf{s}_i := \boldsymbol{\Lambda}(\mathbf{Z}_i + \mathbf{b}) \in \mathbb{R}^K$ across
the inner $j$-loop, updating it incrementally whenever $Z_{ij}$ changes:
$\mathbf{s}_i \leftarrow \mathbf{s}_i + \delta \cdot \boldsymbol{\Lambda}_{:,j}$
with $\delta = Z_{ij}^{\text{new}} - Z_{ij}^{\text{old}}$. Then (6.2) is
implemented as
$$
\mathbb{E}(Z_{ij} \mid \cdots) \;=\; Z_{ij} - \frac{s_{ij}}{\Lambda_{jj}},
$$
which is $\mathcal{O}(K)$ per coordinate. Total cost per Gibbs iteration is
$\mathcal{O}(NK^2)$. This is the same Geweke sweep used by Klotzke & Fox
(2019b) Algorithm 1.

### 6.2 Item difficulties $\mathbf{b}$ — conjugate Gaussian

Treat $\mathbf{Z}$ as observed Gaussian outcomes:
$\mathbf{Z}_i \mid \mathbf{b}, \boldsymbol{\Sigma} \sim \mathcal{N}_K(-\mathbf{b}, \boldsymbol{\Sigma})$. The log
likelihood expanded in $\mathbf{b}$ is
$$
\sum_i \log \mathcal{N}(\mathbf{Z}_i \mid -\mathbf{b}, \boldsymbol{\Sigma})
  \;=\; -\tfrac{1}{2}\bigl[ N \, \mathbf{b}^{\!\top}\boldsymbol{\Lambda} \mathbf{b} + 2 \mathbf{b}^{\!\top}\boldsymbol{\Lambda} \textstyle\sum_i \mathbf{Z}_i \bigr] + \mathrm{const}.
$$
Combined with the Gaussian prior (5.1) this is a quadratic-linear form in
$\mathbf{b}$ giving a Gaussian conditional:
$$
\mathbf{b} \mid \mathbf{Z}, \boldsymbol{\Sigma} \;\sim\; \mathcal{N}_K(\mathbf{m}_b, \mathbf{V}_b),
\quad
\mathbf{V}_b^{-1} = N\,\boldsymbol{\Lambda} + \tau_b^{-2}\mathbf{I},
\quad
\mathbf{m}_b = \mathbf{V}_b\!\left(\tau_b^{-2}\mu_b\mathbf{1} - \boldsymbol{\Lambda} \textstyle\sum_i \mathbf{Z}_i\right).
\tag{6.3}
$$
We draw via a Cholesky factor of $\mathbf{V}_b$. Cost: $\mathcal{O}(K^3)$
per iteration.

### 6.3 Covariance components $\theta_t$ — proof of Proposition 5.1

This is the centrepiece. Let $\mathbf{R}_i := \mathbf{Z}_i + \mathbf{b}$ so that
$\mathbf{R}_i \sim \mathcal{N}_K(\mathbf{0}, \boldsymbol{\Sigma})$. We isolate the dependence on
$\theta_t$ by writing $\boldsymbol{\Sigma} = \boldsymbol{\Sigma}_{-t} + \theta_t \mathbf{u}_t \mathbf{u}_t^{\!\top}$.

By (2.1) and (2.2),
$$
\log\det\boldsymbol{\Sigma} = \log\det\boldsymbol{\Sigma}_{-t} + \log(1 + \theta_t c_t),
\quad
\boldsymbol{\Sigma}^{-1} = \boldsymbol{\Sigma}_{-t}^{-1} - \frac{\theta_t}{1 + \theta_t c_t}\,\mathbf{v}_t\mathbf{v}_t^{\!\top},
$$
where $\mathbf{v}_t := \boldsymbol{\Sigma}_{-t}^{-1}\mathbf{u}_t$ and $c_t = \mathbf{u}_t^{\!\top}\mathbf{v}_t$. Then
$$
\mathbf{R}_i^{\!\top}\boldsymbol{\Sigma}^{-1}\mathbf{R}_i
   \;=\; \mathbf{R}_i^{\!\top}\boldsymbol{\Sigma}_{-t}^{-1}\mathbf{R}_i \;-\; \frac{\theta_t}{1 + \theta_t c_t}\bigl(\mathbf{v}_t^{\!\top}\mathbf{R}_i\bigr)^2.
$$
Summing over $i$ and dropping terms that do not depend on $\theta_t$,
$$
\log p(\mathbf{R} \mid \theta_t, \cdot)
   \;\overset{\theta_t}{=}\;
   -\tfrac{N}{2}\log(1 + \theta_t c_t)
   \;+\; \frac{\theta_t}{2(1 + \theta_t c_t)} \, Q_t,
\tag{6.4}
$$
with $Q_t := \sum_i (\mathbf{v}_t^{\!\top}\mathbf{R}_i)^2$. Now invoke Remark 4.2: under
disjoint binary $\mathbf{u}_t$, $\mathbf{v}_t = \mathbf{u}_t$ and $c_t = k_t$, so
$Q_t = \sum_i (\mathbf{u}_t^{\!\top}\mathbf{R}_i)^2$ and
$$
\log p(\mathbf{R} \mid \theta_t, \cdot)
   \;\overset{\theta_t}{=}\;
   -\tfrac{N}{2}\log(1 + \theta_t k_t)
   \;+\; \frac{\theta_t}{2(1 + \theta_t k_t)} \, Q_t.
\tag{6.5}
$$
Substitute $x = \theta_t + \psi_t$. Then $1 + \theta_t k_t = 1 + k_t (x - \psi_t)$ and
$$
\frac{\theta_t}{1 + \theta_t k_t} \;=\; \frac{x - \psi_t}{1 + k_t(x - \psi_t)}.
$$
**Choose $\psi_t = 1/k_t$ exactly.** Then $1 + k_t(x - \psi_t) = k_t \, x$ and
$$
\frac{\theta_t}{1 + \theta_t k_t} \;=\; \frac{x - 1/k_t}{k_t \, x} \;=\; \frac{1}{k_t} - \frac{1}{k_t^2 \, x}.
$$
The first term does not depend on $x$ and is absorbed into the constant.
Substituting back into (6.5),
$$
\log p(\mathbf{R} \mid \theta_t, \cdot)
   \;\overset{x}{=}\;
   -\tfrac{N}{2}\log(k_t \, x) - \frac{Q_t}{2 \, k_t^2 \, x}
   \;\overset{x}{=}\;
   -\tfrac{N}{2}\log x - \frac{Q_t}{2 \, k_t^2 \, x}.
\tag{6.6}
$$
Adding the log-prior $-\bigl(\alpha_0 + 1\bigr)\log x - \beta_0 / x$ and
collecting,
$$
\log p(\theta_t \mid \mathbf{R}, \boldsymbol{\theta}_{-t})
  \;\overset{x}{=}\;
  -\!\Bigl(\alpha_0 + \tfrac{N}{2} + 1\Bigr) \log x
  \;-\; \frac{\beta_0 + Q_t/(2 k_t^2)}{x}.
\tag{6.7}
$$
This is the kernel of an inverse-gamma density in $x$. Recalling the
truncation $\theta_t > -1/k_t \iff x > 0$, we have
$$
\boxed{\;\;
x \mid \mathbf{R}, \boldsymbol{\theta}_{-t} \;\sim\; \mathrm{InvGamma}\!\bigl(\alpha_0 + N/2, \; \beta_0 + Q_t/(2 k_t^2)\bigr),
\quad \theta_t = x - 1/k_t.
\;\;}
\tag{6.8}
$$
This proves Proposition 5.1.

**Remark 6.1** (*Why $\psi_t = 1/k_t$ is essential*). For any other shift,
$1 + k_t(x - \psi_t) \neq k_t x$ and the log-likelihood retains a term
$-(N/2)\log(1 + k_t x - k_t \psi_t)$ that is *not* affine in $\log x$. The
conditional then is no longer a simple inverse-gamma in $x$, and the
"truncated shifted IG" form (5.2) is no longer reproduced by the
posterior. In practice this means: (i) the implementation must use
$\psi_t = 1/k_t$, and (ii) the implemented Gibbs sample for $\theta_t$
is exactly (6.8), not (5.2).

**Remark 6.2** (*General $\mathbf{u}_t$*). For non-disjoint or non-binary
$\mathbf{u}_t$, equation (6.4) still holds but $c_t$ depends on the other
$\boldsymbol{\theta}_{-t}$ through $\mathbf{v}_t = \boldsymbol{\Sigma}_{-t}^{-1}\mathbf{u}_t$. The
"canonical" shift then becomes $\psi_t^{(\mathrm{iter})} = 1/c_t$, which
varies *across iterations*. This is the route taken implicitly by
Klotzke & Fox (2019a) Eq. 11; it requires recomputing the shift each
iteration but otherwise preserves the closed-form IG draw.

### 6.4 Putting it together

A single Gibbs sweep performs:

1. Form $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}$ from current $(\boldsymbol{\sigma}^2,
   \boldsymbol{\theta})$ via Woodbury, $\mathcal{O}(K^2 T)$ for $T \ll K$.
2. Update $\mathbf{Z}$ by Geweke sweep, $\mathcal{O}(NK^2)$.
3. Update $\mathbf{b}$ by (6.3), $\mathcal{O}(K^3)$.
4. For each $t$, compute $Q_t = \sum_i (\mathbf{u}_t^{\!\top}\mathbf{R}_i)^2$ and draw
   $\theta_t$ from (6.8), $\mathcal{O}(NK)$.

Total per-iteration cost: $\mathcal{O}(NK^2 + K^3)$. For the TIMSS-scale
problem $(N, K) \approx (500, 30)$ this is $\sim 5 \cdot 10^5$ flops per
iteration — milliseconds, with the constant dominated by the latent-utility
sweep.

---

## 7. Mean-field variational inference

The Gibbs sampler is efficient but still produces only correlated samples.
For applications that need posterior quantities at many parameter
settings — e.g. cross-validation, model selection over testlet
specifications, or large-scale screening across countries in TIMSS — a
deterministic approximation can be useful. We derive the mean-field CAVI
that is implemented in `BCSModels.jl::cavi_irt_bcsm` / `cavi_testlet_bcsm`.

### 7.1 Variational family

We factorise the posterior as
$$
q(\mathbf{Z}, \mathbf{b}, \boldsymbol{\theta}) \;=\;
\prod_{i,j} q(Z_{ij}) \cdot \prod_j q(b_j) \cdot \prod_t q(\theta_t),
\tag{7.1}
$$
with each marginal in a known family:
$$
\begin{aligned}
q(Z_{ij}) &\;=\; \mathcal{TN}\!\bigl(m_{ij}, v_{ij}, \,\mathrm{sign}_{ij}\bigr) && \text{(truncated normal)},\\
q(b_j)    &\;=\; \mathcal{N}(m_{b,j}, v_{b,j}),\\
q(\theta_t) &\;=\; \mathcal{IG}\!\bigl(\tilde\alpha_t, \tilde\beta_t\bigr) \text{ on } x_t = \theta_t + 1/k_t > 0.
\end{aligned}
$$
The half-line direction $\mathrm{sign}_{ij} = 2 Y_{ij} - 1$ is fixed by the
data; only the location $m_{ij}$ and scale $v_{ij}$ are variational.

For each variable, the (untruncated) coordinate-ascent update reads
$$
\log q_j^*(z_j) \;\propto\; \mathbb{E}_{q_{-j}}\!\bigl[\log p(\mathbf{Y}, \mathbf{Z}, \mathbf{b}, \boldsymbol{\theta})\bigr]
$$
(Bishop 2006, §10.1; Blei et al. 2017). Conditional conjugacy under the
mean-field factorisation gives closed-form updates for each block.

### 7.2 Update for $q(Z_{ij})$

Conditional log joint, viewed as a function of $Z_{ij}$ alone, is
$$
\log p(\mathbf{Z}, \mathbf{b}, \boldsymbol{\theta}) \;\overset{Z_{ij}}{=}\;
-\tfrac{1}{2}\Lambda_{jj} Z_{ij}^2 \;-\; Z_{ij} \!\sum_{k\neq j} \Lambda_{jk}(Z_{ik} + b_k) \;-\; Z_{ij}\Lambda_{jj} b_j.
$$
Taking expectations under $q_{-Z_{ij}}$, *and* replacing
$\mathbb{E}_q[\boldsymbol{\Lambda}]$ by $\hat{\boldsymbol{\Lambda}} := \boldsymbol{\Sigma}^{-1}\bigl(\mathbb{E}_q[\boldsymbol{\theta}]\bigr)$
(the plug-in approximation; see Section 7.6),
$$
\log q^*(Z_{ij}) \;\overset{Z_{ij}}{=}\;
-\tfrac{1}{2}\hat\Lambda_{jj} Z_{ij}^2 \;+\; Z_{ij}\!\left[-\hat\Lambda_{jj} m_{b,j} - \!\!\sum_{k\neq j}\hat\Lambda_{jk}\!\bigl(m_{z,ik} + m_{b,k}\bigr)\right].
$$
This is the kernel of a Gaussian, truncated to the half-line determined by
$\mathrm{sign}_{ij}$. The variational parameters are
$$
v_{ij} \;=\; \frac{1}{\hat\Lambda_{jj}}, \qquad
m_{ij} \;=\; m_{z,ij} - v_{ij} \sum_{k}\hat\Lambda_{jk}\bigl(m_{z,ik} + m_{b,k}\bigr) + (m_{z,ij}+m_{b,j}) - m_{z,ij}.
$$
(One must be careful: we want the parameter $m_{ij}$ of the untruncated
Gaussian, not the truncated mean. After this update the new truncated mean
$\mathbb{E}_q[Z_{ij}]$ is computed from standard truncated-normal moment
formulas.)

In practice we maintain $\mathbb{E}_q[Z_{ij}] =: \mu_{z,ij}$ and
$\mathrm{Var}_q[Z_{ij}]$. Letting $\alpha_{ij} = -m_{ij}/\sqrt{v_{ij}}$,
$$
\mu_{z,ij} \;=\; m_{ij} \;+\; \mathrm{sign}_{ij} \cdot \sqrt{v_{ij}}\, \lambda(\alpha_{ij}, \mathrm{sign}_{ij}),
$$
where $\lambda$ is the inverse-Mills ratio with sign convention
$$
\lambda(\alpha, +1) = \frac{\phi(\alpha)}{1-\Phi(\alpha)}, \quad
\lambda(\alpha, -1) = \frac{-\phi(\alpha)}{\Phi(\alpha)},
$$
and the truncated variance
$$
\mathrm{Var}_q(Z_{ij}) \;=\; v_{ij}\bigl(1 - \lambda(\lambda - \alpha)\bigr).
$$
See Greene (2003, §22.8) for the standard formulas.

### 7.3 Update for $q(b_j)$

The conditional log joint, as a function of $b_j$ only, is
$$
\log p(\cdots) \;\overset{b_j}{=}\;
-\tfrac{1}{2}\bigl(N\Lambda_{jj} + \tau_b^{-2}\bigr) b_j^2
\;+\; b_j\!\left[\tau_b^{-2}\mu_b - \Lambda_{jj}\!\sum_i Z_{ij} - \!\!\sum_{k\neq j}\!\Lambda_{jk}\!\sum_i(Z_{ik}+b_k)\right].
$$
Taking expectations under $q_{-b_j}$ (plug-in $\hat{\boldsymbol{\Lambda}}$):
$$
v_{b,j} \;=\; \frac{1}{N\hat\Lambda_{jj} + \tau_b^{-2}},
\quad
m_{b,j} \;=\; v_{b,j}\!\left[\tau_b^{-2}\mu_b \;-\; N\!\!\sum_{k}\!\hat\Lambda_{jk}\bar\mu_{z,k} \;-\; N\!\!\sum_{k\neq j}\!\hat\Lambda_{jk} m_{b,k}\right],
\tag{7.2}
$$
where $\bar\mu_{z,k} = \frac{1}{N}\sum_i \mu_{z,ik}$.

### 7.4 Update for $q(\theta_t)$ — closed form via Proposition 5.1

The conditional log joint as a function of $\theta_t$ is (6.5) plus the
log-prior (5.2). The CAVI update replaces sufficient statistics by their
$q$-expectations:
$$
\mathbb{E}_q[Q_t] \;=\; \mathbb{E}_q\!\Bigl[\sum_i \bigl(\mathbf{u}_t^{\!\top}(\mathbf{Z}_i + \mathbf{b})\bigr)^2\Bigr].
$$
Under the mean-field factorisation, $Z_{ij}$ and $b_j$ are independent
across both indices and across each other, so for each $i$,
$$
\mathbb{E}_q\!\left[\Bigl(\sum_{j: u_{t,j}=1}(Z_{ij}+b_j)\Bigr)^2\right] \;=\;
\Bigl(\sum_{j: u_{t,j}=1}(\mu_{z,ij} + m_{b,j})\Bigr)^2
\;+\; \sum_{j: u_{t,j}=1}\bigl(\mathrm{Var}_q[Z_{ij}] + v_{b,j}\bigr).
\tag{7.3}
$$
The first term is the squared posterior-mean sum; the second is the sum of
marginal variances. Substituting into (6.7) and reading off,
$$
\boxed{\;\;
q(\theta_t) \;=\; \mathcal{IG}\!\bigl(\tilde\alpha_t, \tilde\beta_t\bigr) \text{ on } x_t > 0,
\;\; \tilde\alpha_t = \alpha_0 + \tfrac{N}{2}, \;\;
\tilde\beta_t = \beta_0 + \tfrac{1}{2k_t^2}\mathbb{E}_q[Q_t].
\;\;}
\tag{7.4}
$$
The variational moments follow from the IG moment formulas
($\mathbb{E}[x] = \tilde\beta_t / (\tilde\alpha_t - 1)$ for
$\tilde\alpha_t > 1$, etc.):
$$
\mathbb{E}_q[\theta_t] \;=\; \frac{\tilde\beta_t}{\tilde\alpha_t - 1} - \frac{1}{k_t}, \quad
\mathrm{Var}_q[\theta_t] \;=\; \frac{\tilde\beta_t^2}{(\tilde\alpha_t - 1)^2 (\tilde\alpha_t - 2)}.
\tag{7.5}
$$

### 7.5 ELBO

The evidence lower bound is
$$
\mathcal{L}(q) \;=\; \mathbb{E}_q\!\bigl[\log p(\mathbf{Y}, \mathbf{Z}, \mathbf{b}, \boldsymbol{\theta})\bigr] \;-\; \mathbb{E}_q[\log q].
$$
Writing $\boldsymbol{\mu}_{Z,i} + \mathbf{m}_b = \mathbb{E}_q[\mathbf{Z}_i + \mathbf{b}]$ and using
$\mathrm{tr}(\boldsymbol{\Lambda}\mathrm{Cov}_q(\mathbf{Z}_i + \mathbf{b})) = \sum_j \Lambda_{jj}(\mathrm{Var}_q[Z_{ij}] + v_{b,j})$
(from mean-field independence),
$$
\mathbb{E}_q[\log p(\mathbf{Z} \mid \mathbf{b}, \boldsymbol{\theta})]
\;=\; -\tfrac{N}{2}\mathbb{E}_q[\log\det\boldsymbol{\Sigma}]
\;-\; \tfrac{1}{2}\sum_i\!\left[(\boldsymbol{\mu}_{Z,i}+\mathbf{m}_b)^{\!\top}\hat{\boldsymbol{\Lambda}}(\boldsymbol{\mu}_{Z,i}+\mathbf{m}_b) + \sum_j\hat\Lambda_{jj}(\mathrm{Var}_q[Z_{ij}] + v_{b,j})\right].
$$
The other components $\mathbb{E}_q[\log p(\mathbf{b})]$, $\mathbb{E}_q[\log p(\theta_t)]$,
$\mathbb{E}_q[\log q(\cdot)]$ are standard. Implementation details are in
`src/vb/cavi_irt.jl::compute_elbo_irt`. We use the ELBO only for
convergence monitoring; the absolute value depends on omitted additive
constants.

### 7.6 The plug-in approximation and its consequences

In (7.2) and the corresponding update for $q(Z_{ij})$, we replaced
$\mathbb{E}_q[\boldsymbol{\Lambda}]$ by $\hat{\boldsymbol{\Lambda}} := \boldsymbol{\Lambda}(\mathbb{E}_q[\boldsymbol{\theta}])$.
This is the standard mean-field expedient used by Wand (2017) for
inverse-covariance terms that arise in fragments of the joint log density.
The error is
$$
\mathbb{E}_q[\Lambda_{jk}] - \hat\Lambda_{jk} \;=\; \mathcal{O}\!\bigl(\mathrm{Var}_q[\theta_t]/\Sigma_{jj}^2\bigr),
$$
which is small for moderate posterior concentrations. The two consequences
worth flagging:

1. **CAVI is no longer guaranteed to be ELBO-monotone iteration by
   iteration.** Over many iterations the ELBO still increases, but
   monotonicity at every step requires either an exact $\mathbb{E}_q[\boldsymbol{\Lambda}]$
   or a Gaussian-quadrature expansion. In our convergence diagnostics we
   monitor the ELBO trajectory and the relative parameter change, and
   declare convergence based on the latter.

2. **Posterior under-dispersion is amplified.** The plug-in $\hat\Lambda_{jj}$
   underestimates $\mathbb{E}_q[\Lambda_{jj}]$ for concave $\theta_t \mapsto \Lambda_{jj}$,
   tightening $v_{ij}$ and $v_{b,j}$ relative to the true marginal. This
   compounds with the structural bias discussed in §7.7.

### 7.7 Why mean-field VB under-estimates $\theta_t$

The variational expectation (7.3) breaks the posterior covariance
$\mathrm{Cov}_q[Z_{ij}, Z_{ik}] = 0$ for $j \neq k$ by construction. Under
the true posterior, the latent utilities are positively correlated within
testlet $t$ when $\theta_t > 0$. That positive correlation contributes a
*non-trivial covariance term* to
$\mathbb{E}\bigl[(\sum_{j \in \mathcal{T}_t}(Z_{ij}+b_j))^2\bigr]$, which the
mean-field $q$ omits. The consequence is
$\mathbb{E}_q[Q_t] < \mathbb{E}_{\text{true}}[Q_t]$, so $\tilde\beta_t$ from
(7.4) is too small, so $\mathbb{E}_q[\theta_t]$ from (7.5) is too small.
The bias grows with $\theta_t$ and with $k_t$ (more cross-terms missed).
This matches the empirical finding in
[`docs/mcmc_vs_vb.md`](mcmc_vs_vb.md) of a 30–45% under-estimation in the
multi-layer testlet setting.

A structured variational family $q(\mathbf{Z}_i) = \mathcal{N}_K(\boldsymbol{\mu}_i, \mathbf{S}_i)$
with $\mathbf{S}_i$ parameterised by the same BCSM additive structure
preserves the within-testlet correlation and restores the consistency of
$\mathbb{E}_q[Q_t]$. We treat this as the principal direction for future
work; see Section 10.

---

## 8. Identifiability

The probit BCSM is identified under the same constraints as the
multi-group probit IRT model:

1. **Scale**: $\sigma_j^2 \equiv 1$ for all $j$. This eliminates the
   indeterminacy $(\mathbf{b}, \mathbf{Z}, \boldsymbol{\Sigma}) \mapsto c(\mathbf{b}, \mathbf{Z}, \boldsymbol{\Sigma})$
   that would otherwise leave the likelihood invariant.

2. **Location**: A proper Gaussian prior (5.1) with $\mu_b = 0$ identifies
   the location of $\mathbf{b}$. (Equivalently, one can fix $b_1 = 0$ and
   give the rest of $\mathbf{b}$ improper priors; the implementations of
   this paper take the prior-based route.)

3. **Sign of $\boldsymbol{\Sigma}$**: The truncation $\theta_t > \mathrm{tr}_t$ from
   Lemma 4.1, together with the proper TSIG prior, identifies the sign of
   $\theta_t$ uniquely. (Without truncation, the formal posterior would
   place mass on covariance matrices that are not PD.)

No rotation is needed because every $\mathbf{u}_t$ is a fixed binary indicator
(known from the testlet design).

---

## 9. Special cases and extensions

### 9.1 Single-layer IRT-BCSM (Fox 2024)

Take $T = 1$ and $\mathbf{u}_1 = \mathbf{1}_K$. Then $k_1 = K$, $\psi_1 = 1/K$,
$\mathrm{tr}_1 = -1/K$, and (6.8) becomes
$$
\theta + \tfrac{1}{K} \mid \mathbf{R} \;\sim\;
\mathrm{InvGamma}\!\bigl(\alpha_0 + N/2, \, \beta_0 + Q/(2K^2)\bigr), \quad Q = \textstyle\sum_i (\mathbf{1}^{\!\top}\mathbf{R}_i)^2.
$$
This is the small-sample IRT model of Fox (2024).

### 9.2 Disjoint Testlet-BCSM (Fox, Wenzel & Klotzke 2021)

Each item $j$ is in exactly one testlet $t = t(j) \in \{1, \dots, T\}$;
$\mathbf{u}_t \in \{0,1\}^K$ is the testlet indicator. By Remark 4.2,
$c_t = k_t$ and the per-layer update is independent of the others. This
is the setting where our Gibbs and CAVI both have the strongest closed
form.

### 9.3 Overlapping or continuous $\mathbf{u}_t$

When the layer indicators are not orthogonal — e.g. when modelling
crossed content × cognitive domain testlets, or a 2PL with general
discrimination $\mathbf{u}_1 = \mathbf{a}$ — the truncation bound becomes
$\mathrm{tr}_t = -1/c_t$ with $c_t = \mathbf{u}_t^{\!\top}\boldsymbol{\Sigma}_{-t}^{-1}\mathbf{u}_t$. The
closed-form proof in §6.3 still goes through with $k_t \to c_t$ and
$\psi_t = 1/c_t$, but now $\psi_t$ changes with $\boldsymbol{\theta}_{-t}$. Two
practical routes:

* **Adaptive shift.** Recompute $\psi_t^{(\mathrm{iter})}$ at each Gibbs
  iteration. This preserves the closed-form draw at the cost of one
  $\mathcal{O}(K^2)$ inverse-times-vector per iteration. Klotzke & Fox
  (2019a) take this route; the package supports it via the more general
  `AdditiveCovariance` interface but does not enable it by default.
* **Fixed shift + Metropolis–Hastings.** Keep $\psi_t = 1/k_t$ as a *prior*
  and add a Metropolis step around (6.5) for $\theta_t$. This is the
  cleanest path if $c_t$ is hard to compute (e.g. very large $K$).

### 9.4 Polytomous responses

Replace the binary truncation with an ordered-probit threshold scheme
$\gamma_{j, c-1} < Z_{ij} \leq \gamma_{j, c}$ for response category
$c \in \{1, \dots, C_j\}$ (Albert & Chib 1993, §3). The latent-utility
sweep is unchanged except for the truncation interval; the $\mathbf{b}$
update absorbs the thresholds (Cowles 1996; Sahu 2002). The $\theta_t$
update is *identical*. So the BCSM kernel transfers verbatim to
generalised partial-credit / graded-response data.

### 9.5 Response times

The original BCSM extension of Klotzke & Fox (2019b) stacks responses
$\mathbf{Y}$ and log response times $\log \mathbf{T}$ into a single $2K$-vector
with an enlarged $\boldsymbol{\Sigma}$, and the latent-utility step is replaced by an
identity step for the (already-Gaussian) RT block. The CAVI in §7
transfers with the modification that $q(\log T_{ij}) = \delta_{\log T_{ij}}$
(observed) so the corresponding factor drops out.

---

## 10. Discussion and limitations

* **Closed-form Bayes for free.** Section 6 makes precise the algebraic
  identity that turns the BCSM Gibbs into a sequence of inverse-gamma /
  Gaussian draws. The choice $\psi_t = 1/c_t$ is not optional: it is the
  unique shift under which the conditional posterior of $\theta_t$ stays
  in the same family as the prior. Implementations that deviate from this
  shift (as our initial implementation did with $\psi_t = 2/k_t$) produce
  chains that absorb at $\theta_t = \mathrm{tr}_t$ — a pathology that the
  literature mentions only obliquely.

* **CAVI is genuinely new for BCSM.** To the best of our knowledge, no
  prior work has derived a closed-form mean-field VI for the BCSM family.
  The CAVI in §7 inherits the conditional-conjugacy structure of the
  Gibbs and produces *exactly the same family* of variational marginals
  as the Gibbs full conditionals. The only approximations are (i)
  mean-field independence across $(Z_{ij}, b_j, \theta_t)$ and (ii) the
  plug-in $\hat{\boldsymbol{\Lambda}}$.

* **The mean-field bias is structural, not numerical.** Section 7.7
  identifies the dropped within-testlet correlation as the source of the
  empirical 30–45% under-estimation of $\theta_t$ in multi-layer settings.
  A block-structured family $q(\mathbf{Z}_i) = \mathcal{N}_K(\boldsymbol{\mu}_i, \mathbf{S}_i)$
  with $\mathbf{S}_i$ in BCSM form preserves that correlation and should
  close most of the gap. This is the natural follow-up methodology paper.

* **Computational comparison.** The Gibbs at $\mathcal{O}(NK^2 + K^3)$ per
  iteration and the CAVI at $\mathcal{O}(NK^2 + K^3)$ per outer iteration
  have the same nominal complexity. CAVI converges in 50–300 outer
  iterations vs. 1000+ Gibbs samples needed for stable posterior summaries,
  so in practice CAVI is 10–20× faster, as we observed in
  [`docs/mcmc_vs_vb.md`](mcmc_vs_vb.md).

* **What this paper does *not* cover.** (i) Polytomous BCSM with ordered
  probit (sketched in §9.4 but no experiments); (ii) BCSM × response
  times à la Klotzke & Fox (2019b); (iii) BCSM for measurement invariance
  (Fox, Koops, Feskens & Beinhauer 2020); (iv) DIF
  (Fox 2026, in press); (v) the structured-VB upgrade discussed above.
  Each is a separate research thread; we believe the cleanest path is to
  publish (this paper's) IRT-BCSM × {Gibbs, CAVI} comparison first, then
  the structured-VB extension as a follow-up.

---

## Appendix A. Algorithmic pseudocode

### Gibbs sampler for disjoint Testlet-BCSM

```
input: Y ∈ {0,1}^{N×K}, testlet_of ∈ {1..T}^K, niter, burnin
state: b ∈ R^K, θ ∈ R^T, Z ∈ R^{N×K}
for it = 1 .. burnin + niter:
  Σ ← I + Σ_t θ_t u_t u_t^⊤        # via additive structure
  Λ ← Σ^{-1}                        # Woodbury
  s_i ← Λ (Z_i + b)   for each i
  for i = 1..N, j = 1..K:
    m ← Z_{ij} - s_{ij} / Λ_{jj}
    Z_{ij}^new ~ TN(m, 1/Λ_{jj}, sign = 2 Y_{ij} - 1)
    s_i ← s_i + (Z_{ij}^new - Z_{ij}) · Λ_{:,j}
    Z_{ij} ← Z_{ij}^new
  V_b ← (NΛ + I/τ_b²)^{-1}
  m_b ← V_b · (μ_b/τ_b² · 1 - Λ · Σ_i Z_i)
  b ~ N(m_b, V_b)
  for t = 1..T:
    Q_t ← Σ_i (Σ_{j: u_{t,j}=1}(Z_{ij} + b_j))²
    x_t ~ InvGamma(α₀ + N/2, β₀ + Q_t/(2 k_t²))
    θ_t ← x_t - 1/k_t
  if it > burnin: store (b, θ)
output: posterior samples
```

### Mean-field CAVI for disjoint Testlet-BCSM

```
input: Y ∈ {0,1}^{N×K}, testlet_of ∈ {1..T}^K, maxiter, tol
state: m_z, v_z ∈ R^{N×K}; m_b, v_b ∈ R^K; m_θ, v_θ ∈ R^T
for outer = 1 .. maxiter:
  Λ̂ ← Σ^{-1}(m_θ)                 # plug-in
  # q(Z_{ij}) — truncated-normal update
  s_i ← Λ̂ (m_z[i,:] + m_b)
  for i = 1..N, j = 1..K:
    v ← 1 / Λ̂_{jj}
    m ← m_z[i,j] - v · s_{ij}
    (μ_new, ν_new) ← truncated_normal_moments(m, v, sign=2Y_{ij}-1)
    update s_i with (μ_new - m_z[i,j]) increment
    m_z[i,j] ← μ_new; v_z[i,j] ← ν_new
  # q(b_j) — Gaussian update
  for j = 1..K:
    v_b[j] ← 1 / (N Λ̂_{jj} + 1/τ_b²)
    m_b[j] ← v_b[j] · [μ_b/τ_b² - N (Λ̂ m̄_z)_j - N Σ_{k≠j} Λ̂_{jk} m_b[k]]
  # q(θ_t) — closed-form IG (Prop. 5.1, Eq. 7.4)
  for t = 1..T:
    k_t ← Σ_j u_{t,j}
    E_Q_t ← Σ_i [(Σ_{j∈t}(m_z[i,j]+m_b[j]))² + Σ_{j∈t}(v_z[i,j]+v_b[j])]
    α̃ ← α₀ + N/2;  β̃ ← β₀ + E_Q_t / (2 k_t²)
    m_θ[t] ← β̃/(α̃-1) - 1/k_t;  v_θ[t] ← β̃²/((α̃-1)²(α̃-2))
  if ‖Δparams‖ < tol: break
output: (m_b, v_b, m_θ, v_θ, m_z, v_z, ELBO trajectory)
```

---

## Appendix B. Comparison with the literature

| Aspect | This document | Klotzke & Fox (2019a) | Fox (2024) | Cho et al. (2021, VB MIRT) |
|---|---|---|---|---|
| Model | Probit BCSM, IRT setting | Same, general process data | 2PL probit, marginalised | 2-parameter MIRT |
| Inference | Gibbs + CAVI | Gibbs | Gibbs | Gaussian VEM |
| Closed-form $\theta_t$ | IG via $\psi_t = 1/c_t$ (Prop. 5.1) | tsIG (Eq. 11) | IG (Eq. 6) | not applicable |
| Plug-in $\hat\boldsymbol{\Lambda}$ in VB | yes (§7.6) | n/a | n/a | yes, equivalent for diagonal Σ |
| Bias diagnosis | structural, §7.7 | n/a | n/a | not discussed |
| Reproducible code | `BCSModels.jl` (this package) | R / Gibbs only | R | MATLAB / VEM only |

The principal contribution of this document, beyond consolidating the
literature, is **the CAVI in §7 and the explicit identification in
Proposition 5.1 of why $\psi_t = 1/k_t$ (rather than the looser
$\psi_t \geq 1/k_t$) is required for a closed-form Gibbs / CAVI**.

---

## References

Albert, J. H., & Chib, S. (1993). Bayesian analysis of binary and
polychotomous response data. *JASA*, **88**(422), 669–679.

Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*.
Springer.

Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational
inference: A review for statisticians. *JASA*, **112**(518), 859–877.

Cho, A. E., Wang, C., Zhang, X., & Xu, G. (2021). Gaussian variational
estimation for multidimensional item response theory. *British Journal of
Mathematical and Statistical Psychology*, **74**, 52–85.

Cowles, M. K. (1996). Accelerating Monte Carlo Markov chain convergence
for cumulative-link generalized linear models. *Statistics and Computing*,
**6**, 101–111.

Fox, J.-P. (2010). *Bayesian Item Response Modeling: Theory and
Applications*. Springer.

Fox, J.-P. (2024). Redefining item response models for small samples.
*Journal of Educational and Behavioral Statistics*, **50**(2), 272–295.

Fox, J.-P. (2026, in press). Bayesian covariance modeling of differential
item functioning. *Psychometrika*.

Fox, J.-P., Koops, J., Feskens, R., & Beinhauer, L. (2020). Bayesian
covariance structure modelling for measurement invariance testing.
*Behaviormetrika*, **47**, 385–410.

Fox, J.-P., Mulder, J., & Sinharay, S. (2017). Bayes factor covariance
testing in item response models. *Psychometrika*, **82**(4), 979–1006.

Fox, J.-P., & Smink, W. A. C. (2021). Assessing an alternative for
"negative variance components". *arXiv:2106.10107*.

Fox, J.-P., Wenzel, J., & Klotzke, K. (2021). The Bayesian covariance
structure model for testlets. *Journal of Educational and Behavioral
Statistics*, **46**(2), 219–243.

Geweke, J. (1991). Efficient simulation from the multivariate normal and
Student-t distributions subject to linear constraints. In *Computing
Science and Statistics: Proceedings of the 23rd Symposium*, 571–578.

Greene, W. H. (2003). *Econometric Analysis* (5th ed.). Prentice Hall.

Klotzke, K., & Fox, J.-P. (2019a). Bayesian covariance structure modeling
of responses and process data. *Frontiers in Psychology*, **10**:1675.

Klotzke, K., & Fox, J.-P. (2019b). Modeling dependence structures for
response times in a Bayesian framework. *Psychometrika*, **84**(3),
649–672.

Mulder, J., & Fox, J.-P. (2013). Bayesian tests on components of the
compound symmetry covariance matrix. *Statistics and Computing*,
**23**(1), 109–122.

Mulder, J., & Fox, J.-P. (2019). Bayes factor testing of multiple
intraclass correlations. *Bayesian Analysis*, **14**(2), 521–552.

Nielsen, N. M., Smink, W. A. C., & Fox, J.-P. (2021). Small and negative
correlations among clustered observations: limitations of the linear
mixed-effects model. *Behaviormetrika*, **48**, 51–77.

Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian inference for
logistic models using Pólya–Gamma latent variables. *JASA*, **108**(504),
1339–1349.

Sahu, S. K. (2002). Bayesian estimation and model choice in item
response models. *Journal of Statistical Computation and Simulation*,
**72**, 217–232.

Wand, M. P. (2017). Fast approximate inference for arbitrarily large
semiparametric regression models via message passing. *JASA*, **112**(517),
137–168.
