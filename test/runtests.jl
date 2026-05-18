using BCSModels
using Test, Random, Statistics, LinearAlgebra

# Code-quality checks (Aqua, JET) are gated by the BCSMODELS_QUALITY_TESTS
# environment variable. CI sets this to "true" (default); locally you can
# `BCSMODELS_QUALITY_TESTS=false julia --project=. -e 'using Pkg; Pkg.test()'`
# to skip the slower static-analysis step.
const RUN_QUALITY_TESTS = get(ENV, "BCSMODELS_QUALITY_TESTS", "true") == "true"

@testset "BCSModels" begin

    if RUN_QUALITY_TESTS
        include("aqua_test.jl")
        include("jet_test.jl")
    end

    @testset "TruncatedShiftedInverseGamma" begin
        rng = MersenneTwister(1)
        d = BCSModels.TruncatedShiftedInverseGamma(3.0, 2.0, 0.5, -0.2)
        samples = [BCSModels.rand_tsig(rng, d) for _ in 1:50_000]
        @test isapprox(mean(samples), BCSModels.mean_tsig(d); atol=0.05)
        @test isapprox(var(samples),  BCSModels.var_tsig(d);  atol=0.10)
        # support: must be > tr
        @test minimum(samples) > d.tr
        # degenerate L = 0 case (ψ = -tr) reduces to standard IG
        d0 = BCSModels.TruncatedShiftedInverseGamma(3.0, 2.0, 0.2, -0.2)
        sx = [BCSModels.rand_tsig(rng, d0) for _ in 1:20_000]
        @test minimum(sx) > -0.2 - 1e-9
        @test isapprox(mean(sx), 2.0 / (3.0 - 1) - 0.2; atol=0.05)
    end

    @testset "AdditiveCovariance" begin
        K = 8
        U = ones(K, 1)
        Σ = BCSModels.AdditiveCovariance(ones(K), [0.4], U)
        M = BCSModels.build_covariance(Σ)
        @test isposdef(M)
        @test isapprox(M * BCSModels.inv_covariance(Σ), I(K); atol=1e-9)
        # diagonal of inverse
        d_inv = BCSModels.diag_inv_covariance(Σ)
        @test isapprox(d_inv, diag(BCSModels.inv_covariance(Σ)); atol=1e-9)
        # truncation bound for single layer u = 1_K, Σ₀ = I is -1/K
        Σ_dummy = BCSModels.AdditiveCovariance(ones(K), [0.0], U)
        # truncation_bound expects to "remove" layer t — here layer 1 has θ=0
        @test isapprox(BCSModels.truncation_bound(Σ_dummy, 1), -1.0 / K; atol=1e-9)
    end

    @testset "IRT-BCSM Gibbs recovers θ on simulated data" begin
        rng = MersenneTwister(2025)
        N, K = 800, 10
        θ_true = 0.4
        Y, info = BCSModels.simulate_irt_bcsm(rng, N, K; θ_true=θ_true, σ_b=1.0)
        model = BCSModels.IRTBCSM(K=K)
        res = BCSModels.gibbs_irt_bcsm(Y, model; niter=1500, burnin=500,
                                  rng=MersenneTwister(99))
        θ_post = mean(vec(res.samples_θ))
        θ_lo = quantile(vec(res.samples_θ), 0.025)
        θ_hi = quantile(vec(res.samples_θ), 0.975)
        @test θ_lo ≤ θ_true ≤ θ_hi          # 95% CI covers truth
        @test abs(θ_post - θ_true) < 0.10    # not wildly off
        b_mean = vec(mean(res.samples_b, dims=1))
        @test cor(b_mean, info.b) > 0.95     # b recovered
    end

    @testset "IRT-BCSM CAVI: convergence and approximate recovery" begin
        rng = MersenneTwister(11)
        N, K = 800, 10
        Y, info = BCSModels.simulate_irt_bcsm(rng, N, K; θ_true=0.4, σ_b=1.0)
        model = BCSModels.IRTBCSM(K=K)
        vb = BCSModels.cavi_irt_bcsm(Y, model; maxiter=400, tol=1e-8)
        @test vb.converged
        @test cor(vb.m_b, info.b) > 0.95     # b recovered
        @test 0.10 < vb.m_θ[1] < 0.50        # θ in plausible range
        # ELBO improves overall (the plug-in E_q[Λ] ≈ Λ̂ makes the update
        # only an approximate CAVI, so monotonicity is not guaranteed
        # iteration-by-iteration, but the late ELBO must exceed the early one).
        @test vb.elbo[end] > vb.elbo[1]
    end

    @testset "Testlet-BCSM Gibbs recovers θ per layer" begin
        rng = MersenneTwister(7)
        N, K = 1000, 12
        testlet_of = repeat(1:3, inner=4)
        θ_true = [0.3, 0.5, 0.2]
        Y, info = BCSModels.simulate_testlet_bcsm(rng, N, K;
                                              testlet_of=testlet_of,
                                              θ_true=θ_true)
        model = BCSModels.TestletBCSM(K=K, testlet_of=testlet_of)
        res = BCSModels.gibbs_testlet_bcsm(Y, model; niter=1500, burnin=500,
                                      rng=MersenneTwister(101))
        θ_means = vec(mean(res.samples_θ, dims=1))
        @test all(abs.(θ_means .- θ_true) .< 0.12)
    end

end
