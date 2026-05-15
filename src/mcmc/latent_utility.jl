# Latent utility (Albert–Chib) Gibbs scan for BCSM.
#
# Implements Geweke (1991) one-at-a-time updates of `z_{ij} | z_{i,-j}, y, b, Σ`.
# Given Λ = Σ^{-1} (precomputed once per outer iteration), the conditional
# distribution is `N(z_{ij} - Λ_{jj}^{-1} s_{ij}, Λ_{jj}^{-1})`, truncated to
# the half-line determined by y_{ij}, where s_i = Λ (z_i + b) is maintained
# incrementally as items are scanned. Cost per outer iteration: O(N K²) using
# explicit Λ — acceptable for the K ≈ 30 sizes typical of TIMSS booklets.

"""
    sample_latent_z!(rng, Z, Y, b, Λ; sign_map=_default_signs)

In-place update of the N × K latent-utility matrix Z given binary responses Y,
item intercepts b, and precomputed precision Λ = Σ^{-1}.
"""
function sample_latent_z!(rng::AbstractRNG,
                          Z::AbstractMatrix{Float64},
                          Y::AbstractMatrix{<:Integer},
                          b::AbstractVector{Float64},
                          Λ::AbstractMatrix{Float64})
    N, K = size(Z)
    @assert size(Y) == (N, K)
    @assert length(b) == K
    @assert size(Λ) == (K, K)
    v = 1.0 ./ diag(Λ)         # conditional variances, length K
    s = Λ * (Z' .+ b)          # K × N: s[:,i] = Λ (z_i + b)
    @inbounds for i in 1:N
        for j in 1:K
            mij = Z[i, j] - v[j] * s[j, i]
            sgn = Y[i, j] == 1 ? 1 : -1
            znew = truncnorm_rand(rng, mij, v[j], sgn)
            δ = znew - Z[i, j]
            Z[i, j] = znew
            # update s[:,i] with the new contribution of column j
            for k in 1:K
                s[k, i] += Λ[k, j] * δ
            end
        end
    end
    return Z
end
