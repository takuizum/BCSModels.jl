# Additive covariance Σ = diag(σ²) + Σ_t θ_t u_t u_tᵀ as used by BCSM
# (Klotzke & Fox 2019a, Eq. 1). The K × K representation is small for the
# IRT contexts we target (TIMSS booklets are K ≈ 30, with at most a handful of
# layers T), so we keep things explicit rather than implementing a fully
# Woodbury-only data structure. The truncation bound and Σ^{-1} are computed
# via Woodbury when T < K to avoid forming a dense inverse.

"""
    AdditiveCovariance

A BCSM additive covariance with diagonal base `σ²` and rank-1 layers
`θ_t u_t u_tᵀ`. Stored fields: `σ²::Vector{Float64}` (length K), `θ::Vector`
(length T), `U::Matrix{Float64}` (K × T, each column the binary indicator `u_t`).
"""
struct AdditiveCovariance
    σ²::Vector{Float64}
    θ::Vector{Float64}
    U::Matrix{Float64}   # K × T, each column is u_t (binary or general)
    function AdditiveCovariance(σ², θ, U)
        K = length(σ²)
        T = length(θ)
        size(U) == (K, T) || throw(DimensionMismatch(
            "U must be ($K, $T), got $(size(U))"))
        all(>(0), σ²) || throw(ArgumentError("σ² entries must be positive"))
        return new(collect(float.(σ²)), collect(float.(θ)), Matrix{Float64}(U))
    end
end

Base.size(Σ::AdditiveCovariance) = (length(Σ.σ²), length(Σ.σ²))
n_layers(Σ::AdditiveCovariance) = length(Σ.θ)
n_items(Σ::AdditiveCovariance) = length(Σ.σ²)

"""
    build_covariance(Σ::AdditiveCovariance) -> Matrix{Float64}

Materialise the K × K covariance.
"""
function build_covariance(Σ::AdditiveCovariance)
    K = n_items(Σ)
    M = Matrix{Float64}(undef, K, K)
    @inbounds for j in 1:K, k in 1:K
        s = 0.0
        for t in 1:n_layers(Σ)
            s += Σ.θ[t] * Σ.U[j, t] * Σ.U[k, t]
        end
        M[j, k] = s + (j == k ? Σ.σ²[j] : 0.0)
    end
    return M
end

"""
    inv_covariance(Σ::AdditiveCovariance) -> Matrix{Float64}

Compute Σ^{-1} by Woodbury,
    Σ^{-1} = D^{-1} - D^{-1} U M^{-1} Uᵀ D^{-1},
    M = diag(1/θ) + Uᵀ D^{-1} U,
when all `θ_t ≠ 0`; falls back to direct inversion if any layer has zero or
negative `θ` that makes `1/θ` ill-defined. The fallback handles the
identifiability degenerate case during iteration.
"""
function inv_covariance(Σ::AdditiveCovariance)
    K, T = n_items(Σ), n_layers(Σ)
    Dinv = Diagonal(1 ./ Σ.σ²)
    if T == 0
        return Matrix(Dinv)
    end
    if all(!iszero, Σ.θ)
        DiU = Dinv * Σ.U                       # K × T
        M = Diagonal(1 ./ Σ.θ) + Σ.U' * DiU    # T × T
        # symmetrise to suppress accumulated FP asymmetry before solve
        Msym = Symmetric((M + M') / 2)
        return Matrix(Dinv) - DiU * (Msym \ DiU')
    else
        return inv(Symmetric(build_covariance(Σ)))
    end
end

"""
    diag_inv_covariance(Σ::AdditiveCovariance) -> Vector{Float64}

Diagonal of `Σ^{-1}` only (size K), avoiding materialising the full inverse
when the caller only needs `(Σ^{-1})_{jj}` for j = 1, …, K. Uses Woodbury.
"""
function diag_inv_covariance(Σ::AdditiveCovariance)
    K, T = n_items(Σ), n_layers(Σ)
    if T == 0 || all(iszero, Σ.θ)
        return 1 ./ Σ.σ²
    end
    if all(!iszero, Σ.θ)
        Dinv = 1 ./ Σ.σ²
        DiU = Σ.U .* Dinv
        M = Diagonal(1 ./ Σ.θ) + Σ.U' * DiU
        Msym = Symmetric((M + M') / 2)
        # diag(DiU * M^{-1} * DiU') = rowsum(DiU .* (M^{-1} * DiU'))
        S = Msym \ DiU'        # T × K
        d = vec(sum(DiU .* S', dims=2))
        return Dinv .- d
    else
        return diag(inv(Symmetric(build_covariance(Σ))))
    end
end

"""
    truncation_bound(Σ::AdditiveCovariance, t::Int) -> Float64

Compute `tr_t = -1 / (u_tᵀ Σ_{-t}^{-1} u_t)` where Σ_{-t} is the additive
covariance with layer `t` zeroed (Klotzke & Fox 2019a, Eq. 6). Adding the
layer `θ_t u_t u_tᵀ` keeps `Σ` positive definite iff `θ_t > tr_t`.
"""
function truncation_bound(Σ::AdditiveCovariance, t::Int)
    θ_other = copy(Σ.θ); θ_other[t] = 0.0
    Σ_other = AdditiveCovariance(Σ.σ², θ_other, Σ.U)
    Λ = inv_covariance(Σ_other)
    u = @view Σ.U[:, t]
    quad = dot(u, Λ * u)
    return -1.0 / max(quad, eps())
end

"""
    sherman_morrison_inv_diag(Λ::AbstractMatrix, θ::Real, u::AbstractVector)
        -> (Λ_new, diag_new)

Apply a rank-1 Sherman–Morrison update to an explicit inverse `Λ ≈ Σ^{-1}`,
returning the new inverse and its diagonal. Only used in tests / utilities;
production code uses Woodbury via `inv_covariance` for clarity.
"""
function sherman_morrison_inv_diag(Λ::AbstractMatrix, θ::Real, u::AbstractVector)
    Λu = Λ * u
    denom = 1 + θ * dot(u, Λu)
    Λ_new = Λ - (θ / denom) * (Λu * Λu')
    return Λ_new, diag(Λ_new)
end
