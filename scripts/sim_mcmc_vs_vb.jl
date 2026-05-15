# Simulation experiment: MCMC (Gibbs) vs. VB (mean-field CAVI) on BCSM models.
#
# Grid:
#   * single-layer IRT-BCSM over (N, K, θ_true)
#   * Testlet-BCSM with 3 testlets over (N, K, θ_true per testlet)
#
# For each cell we run R replicates and record:
#   * point estimate of θ (Gibbs posterior mean; CAVI variational mean)
#   * 95% interval coverage of θ_true
#   * interval width
#   * wall time
#   * correlation of item-difficulty estimate with truth
#
# Usage:
#   julia --project=. scripts/sim_mcmc_vs_vb.jl [--reps R] [--out path]
#
# Outputs a CSV with one row per (model, cell, replicate, method).

using BCSM
using Random, Statistics, LinearAlgebra, Printf, Dates

# -------------------- argument parsing --------------------
function parse_args(args)
    out = "results/sim_mcmc_vs_vb_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv"
    reps = 20
    niter = 1500; burnin = 500
    i = 1
    while i ≤ length(args)
        if args[i] == "--reps"
            reps = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--out"
            out = args[i+1]; i += 2
        elseif args[i] == "--niter"
            niter = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--burnin"
            burnin = parse(Int, args[i+1]); i += 2
        else
            error("unknown argument $(args[i])")
        end
    end
    return (out=out, reps=reps, niter=niter, burnin=burnin)
end

# -------------------- IRT-BCSM grid --------------------
function run_irt_cell(N, K, θ_true, rep, niter, burnin)
    rng_sim = MersenneTwister(rep)
    Y, info = BCSM.simulate_irt_bcsm(rng_sim, N, K; θ_true=θ_true, σ_b=1.0)
    model = BCSM.IRTBCSM(K=K)

    # Gibbs
    g = BCSM.gibbs_irt_bcsm(Y, model; niter=niter, burnin=burnin,
                            rng=MersenneTwister(rep + 10_000))
    θ_chain = vec(g.samples_θ)
    g_mean = mean(θ_chain)
    g_lo, g_hi = quantile(θ_chain, [0.025, 0.975])
    g_cover = g_lo ≤ θ_true ≤ g_hi
    g_b = vec(mean(g.samples_b, dims=1))
    g_b_corr = cor(g_b, info.b)
    g_b_rmse = sqrt(mean((g_b .- info.b) .^ 2))

    # CAVI
    vb = BCSM.cavi_irt_bcsm(Y, model; maxiter=400, tol=1e-7)
    vb_sd_θ = sqrt(max(vb.v_θ[1], 0))
    vb_lo = vb.m_θ[1] - 1.96 * vb_sd_θ
    vb_hi = vb.m_θ[1] + 1.96 * vb_sd_θ
    vb_cover = vb_lo ≤ θ_true ≤ vb_hi
    vb_b_corr = cor(vb.m_b, info.b)
    vb_b_rmse = sqrt(mean((vb.m_b .- info.b) .^ 2))

    return [
        ("IRT", N, K, "θ", θ_true, rep, "MCMC",
         g_mean, std(θ_chain), g_lo, g_hi, g_hi - g_lo, Int(g_cover),
         g_b_corr, g_b_rmse, g.elapsed),
        ("IRT", N, K, "θ", θ_true, rep, "VB",
         vb.m_θ[1], vb_sd_θ, vb_lo, vb_hi, vb_hi - vb_lo, Int(vb_cover),
         vb_b_corr, vb_b_rmse, vb.elapsed),
    ]
end

# -------------------- Testlet grid --------------------
function run_testlet_cell(N, K_per, T, θ_true_vec, rep, niter, burnin)
    K = K_per * T
    testlet_of = repeat(1:T, inner=K_per)
    rng_sim = MersenneTwister(rep + 50_000)
    Y, info = BCSM.simulate_testlet_bcsm(rng_sim, N, K;
                                          testlet_of=testlet_of,
                                          θ_true=θ_true_vec,
                                          σ_b=1.0)
    model = BCSM.TestletBCSM(K=K, testlet_of=testlet_of)

    g = BCSM.gibbs_testlet_bcsm(Y, model; niter=niter, burnin=burnin,
                                rng=MersenneTwister(rep + 60_000))
    vb = BCSM.cavi_testlet_bcsm(Y, model; maxiter=400, tol=1e-7)

    rows = Any[]
    for t in 1:T
        chain = g.samples_θ[:, t]
        gm = mean(chain); gs = std(chain)
        glo, ghi = quantile(chain, [0.025, 0.975])
        gcover = glo ≤ θ_true_vec[t] ≤ ghi

        vbsd = sqrt(max(vb.v_θ[t], 0))
        vlo  = vb.m_θ[t] - 1.96 * vbsd
        vhi  = vb.m_θ[t] + 1.96 * vbsd
        vcover = vlo ≤ θ_true_vec[t] ≤ vhi

        push!(rows, ("TLT", N, K, "θ$t", θ_true_vec[t], rep, "MCMC",
                     gm, gs, glo, ghi, ghi - glo, Int(gcover),
                     NaN, NaN, g.elapsed))
        push!(rows, ("TLT", N, K, "θ$t", θ_true_vec[t], rep, "VB",
                     vb.m_θ[t], vbsd, vlo, vhi, vhi - vlo, Int(vcover),
                     NaN, NaN, vb.elapsed))
    end
    return rows
end

# -------------------- main --------------------
function main(args)
    opts = parse_args(args)
    mkpath(dirname(opts.out))
    header = ["model","N","K","param","truth","rep","method",
              "estimate","sd","ci_lo","ci_hi","ci_width","cover",
              "b_corr","b_rmse","elapsed_s"]
    rows_all = Vector{Vector{Any}}()

    irt_cells = [
        (200, 10, 0.20),  (200, 10, 0.40),  (200, 10, 0.60),
        (500, 10, 0.20),  (500, 10, 0.40),  (500, 10, 0.60),
        (1000, 20, 0.20), (1000, 20, 0.40), (1000, 20, 0.60),
    ]
    for (N, K, θt) in irt_cells, rep in 1:opts.reps
        for row in run_irt_cell(N, K, θt, rep, opts.niter, opts.burnin)
            push!(rows_all, collect(row))
        end
        @printf("IRT  N=%d K=%d θ=%.2f rep=%d done\n", N, K, θt, rep)
        flush(stdout)
    end

    testlet_cells = [
        (500, 4, 3, [0.20, 0.40, 0.30]),
        (500, 4, 3, [0.40, 0.20, 0.50]),
        (1000, 5, 4, [0.25, 0.35, 0.45, 0.15]),
    ]
    for (N, kp, T, θv) in testlet_cells, rep in 1:opts.reps
        for row in run_testlet_cell(N, kp, T, θv, rep, opts.niter, opts.burnin)
            push!(rows_all, collect(row))
        end
        @printf("TLT  N=%d K=%d rep=%d done\n", N, kp * T, rep)
        flush(stdout)
    end

    open(opts.out, "w") do io
        println(io, join(header, ","))
        for row in rows_all
            println(io, join(map(string, row), ","))
        end
    end
    @info "wrote results" path=opts.out nrows=length(rows_all)
end

main(ARGS)
