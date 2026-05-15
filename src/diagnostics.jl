# Posterior summary and convergence diagnostics.

"""
    posterior_summary(samples; q=(0.025, 0.5, 0.975)) -> NamedTuple

Compute posterior mean, sd, and quantiles for each column of `samples`
(`niter × P`).
"""
function posterior_summary(samples::AbstractMatrix; q=(0.025, 0.5, 0.975))
    P = size(samples, 2)
    means = vec(mean(samples, dims=1))
    sds   = vec(std(samples, dims=1))
    qs = Matrix{Float64}(undef, P, length(q))
    for p in 1:P
        qs[p, :] = quantile(samples[:, p], collect(q))
    end
    return (mean=means, sd=sds, quantiles=qs, qnames=collect(q))
end

"""
    coverage(samples, truth; level=0.95) -> Vector{Bool}

Indicator vector: was the truth covered by the central credible interval at
the given level? `samples` is `niter × P`, `truth` is a length-P vector.
"""
function coverage(samples::AbstractMatrix, truth::AbstractVector; level::Float64=0.95)
    α = (1 - level) / 2
    P = size(samples, 2)
    res = Vector{Bool}(undef, P)
    for p in 1:P
        lo = quantile(samples[:, p], α)
        hi = quantile(samples[:, p], 1 - α)
        res[p] = lo ≤ truth[p] ≤ hi
    end
    return res
end

"""
    ess_basic(chain) -> Float64

Simple effective sample size from autocovariance (Geyer's initial monotone
sequence). For a length-`n` univariate chain. Returns NaN for constant chains.
"""
function ess_basic(chain::AbstractVector)
    n = length(chain)
    n > 4 || return float(n)
    c = chain .- mean(chain)
    v0 = mean(c .^ 2)
    v0 == 0 && return NaN
    ρ_sum = 0.0
    last_pair = Inf
    k = 1
    while 2k + 1 ≤ n
        ρ1 = mean(c[1:end-(2k-1)] .* c[(2k-1)+1:end]) / v0
        ρ2 = mean(c[1:end-(2k)]   .* c[(2k)+1:end])   / v0
        pair = ρ1 + ρ2
        pair < 0 && break
        pair = min(pair, last_pair)
        ρ_sum += pair
        last_pair = pair
        k += 1
    end
    τ = 1 + 2ρ_sum
    return n / τ
end

"""
    vb_summary(res::VBResult; level=0.95) -> NamedTuple

Build a Gaussian-approximation credible-interval summary from the CAVI output
(uses the Gaussian distribution for `b` and a normal approximation around the
TSIG posterior mean for `θ`).
"""
function vb_summary(res::VBResult; level::Float64=0.95)
    z = StatsFuns.norminvcdf((1 + level) / 2)
    b_lo = res.m_b .- z .* sqrt.(res.v_b)
    b_hi = res.m_b .+ z .* sqrt.(res.v_b)
    θ_lo = res.m_θ .- z .* sqrt.(clamp.(res.v_θ, 0.0, Inf))
    θ_hi = res.m_θ .+ z .* sqrt.(clamp.(res.v_θ, 0.0, Inf))
    return (b_mean=res.m_b, b_sd=sqrt.(res.v_b), b_lo=b_lo, b_hi=b_hi,
            θ_mean=res.m_θ, θ_sd=sqrt.(clamp.(res.v_θ, 0.0, Inf)),
            θ_lo=θ_lo, θ_hi=θ_hi,
            elbo=res.elbo, converged=res.converged, n_iter=res.n_iter,
            elapsed=res.elapsed)
end
