# Parameter-recovery simulation for BCSM.
#
# Vary sample size N and number of items K on a grid. For each cell, draw
# `reps` independent data sets from the data-generating BCSM, fit both the
# Gibbs sampler and the mean-field CAVI, and record how well each method
# recovers:
#   * the covariance component θ  (point estimate, posterior SD, 95 % CI,
#                                  coverage, absolute error)
#   * the item-difficulty vector b (RMSE, correlation, max abs error)
# plus wall time.
#
# Two model families:
#   * IRT-BCSM    (single layer, u = 1_K, θ scalar)
#   * Testlet-BCSM (T testlets, θ vector — reported per layer)
#
# Usage:
#   julia --project=. scripts/param_recovery.jl                    # medium grid
#   julia --project=. scripts/param_recovery.jl --grid small       # quick smoke
#   julia --project=. scripts/param_recovery.jl --grid full --reps 30  # full
#   julia --project=. scripts/param_recovery.jl --out path/to.csv  # custom output
#
# The output CSV has one row per (cell, replicate, method) and is consumed
# by scripts/analyze_recovery.jl to produce the summary tables.

using BCSM
using Random, Statistics, LinearAlgebra, Printf, Dates

# ─────────────────────────── argument parsing ─────────────────────────── #

function parse_args(args)
    grid   = "medium"
    out    = nothing
    reps   = nothing
    niter  = 1500
    burnin = 500
    seed   = 20250515
    i = 1
    while i ≤ length(args)
        a = args[i]
        if     a == "--grid";   grid   = args[i+1]; i += 2
        elseif a == "--reps";   reps   = parse(Int, args[i+1]); i += 2
        elseif a == "--out";    out    = args[i+1]; i += 2
        elseif a == "--niter";  niter  = parse(Int, args[i+1]); i += 2
        elseif a == "--burnin"; burnin = parse(Int, args[i+1]); i += 2
        elseif a == "--seed";   seed   = parse(Int, args[i+1]); i += 2
        else error("unknown argument $a")
        end
    end
    return (grid=grid, out=out, reps=reps, niter=niter, burnin=burnin, seed=seed)
end

# ─────────────────────────── grids ─────────────────────────── #

function design(grid_name, reps_override)
    if grid_name == "small"
        N_grid = [100, 300]
        K_grid = [10, 20]
        θ_grid = [0.4]
        reps   = isnothing(reps_override) ? 5  : reps_override
    elseif grid_name == "medium"
        N_grid = [100, 250, 500, 1000]
        K_grid = [10, 20, 30]
        θ_grid = [0.2, 0.4]
        reps   = isnothing(reps_override) ? 15 : reps_override
    elseif grid_name == "full"
        N_grid = [100, 250, 500, 1000, 2000]
        K_grid = [10, 20, 30, 40]
        θ_grid = [0.1, 0.3, 0.5]
        reps   = isnothing(reps_override) ? 25 : reps_override
    else
        error("unknown grid: $grid_name (use small | medium | full)")
    end
    return (N_grid=N_grid, K_grid=K_grid, θ_grid=θ_grid, reps=reps)
end

# ─────────────────────────── per-cell run ─────────────────────────── #

const HEADER = [
    "model","N","K","θ_layer","θ_true","rep","method",
    # θ recovery
    "θ_est","θ_sd","θ_ci_lo","θ_ci_hi","θ_ci_width","θ_cover","θ_abs_error",
    # b recovery
    "b_rmse","b_corr","b_max_abs_err","b_mean_abs_err","b_cover_rate",
    # diagnostics
    "elapsed_s","n_iter_vb","converged_vb",
]

# Coverage rate of b across the K items, given chain samples.
function b_coverage(samples_b::AbstractMatrix, b_true::AbstractVector)
    K = size(samples_b, 2)
    cov = 0
    for j in 1:K
        lo, hi = quantile(samples_b[:, j], (0.025, 0.975))
        cov += (lo ≤ b_true[j] ≤ hi) ? 1 : 0
    end
    return cov / K
end

# For VB: Gaussian approximation interval for each b_j
function b_coverage_vb(m_b::AbstractVector, v_b::AbstractVector, b_true::AbstractVector)
    K = length(m_b)
    cov = 0
    for j in 1:K
        sd = sqrt(max(v_b[j], 0.0))
        lo, hi = m_b[j] - 1.96 * sd, m_b[j] + 1.96 * sd
        cov += (lo ≤ b_true[j] ≤ hi) ? 1 : 0
    end
    return cov / K
end

function run_irt(N, K, θ_true, rep, niter, burnin, master_seed)
    rng_data = MersenneTwister(hash((:irt, N, K, θ_true, rep, master_seed)))
    Y, info = BCSM.simulate_irt_bcsm(rng_data, N, K; θ_true=θ_true, σ_b=1.0)
    model = BCSM.IRTBCSM(K=K)

    # Gibbs
    rng_g = MersenneTwister(hash((:irt_gibbs, N, K, θ_true, rep, master_seed)))
    g = BCSM.gibbs_irt_bcsm(Y, model; niter=niter, burnin=burnin, rng=rng_g)
    θ_chain = vec(g.samples_θ)
    g_est = mean(θ_chain); g_sd = std(θ_chain)
    g_lo, g_hi = quantile(θ_chain, (0.025, 0.975))
    g_cover = (g_lo ≤ θ_true ≤ g_hi) ? 1 : 0
    g_abs  = abs(g_est - θ_true)
    g_b_post = vec(mean(g.samples_b, dims=1))
    g_b_rmse = sqrt(mean((g_b_post .- info.b) .^ 2))
    g_b_corr = cor(g_b_post, info.b)
    g_b_max  = maximum(abs.(g_b_post .- info.b))
    g_b_mae  = mean(abs.(g_b_post .- info.b))
    g_b_cover = b_coverage(g.samples_b, info.b)

    # CAVI
    vb = BCSM.cavi_irt_bcsm(Y, model; maxiter=400, tol=1e-7)
    v_sd = sqrt(max(vb.v_θ[1], 0))
    v_lo, v_hi = vb.m_θ[1] - 1.96 * v_sd, vb.m_θ[1] + 1.96 * v_sd
    v_cover = (v_lo ≤ θ_true ≤ v_hi) ? 1 : 0
    v_abs   = abs(vb.m_θ[1] - θ_true)
    v_b_rmse = sqrt(mean((vb.m_b .- info.b) .^ 2))
    v_b_corr = cor(vb.m_b, info.b)
    v_b_max  = maximum(abs.(vb.m_b .- info.b))
    v_b_mae  = mean(abs.(vb.m_b .- info.b))
    v_b_cover = b_coverage_vb(vb.m_b, vb.v_b, info.b)

    rows = Vector{Vector{Any}}()
    push!(rows, ["IRT", N, K, "θ", θ_true, rep, "MCMC",
                 g_est, g_sd, g_lo, g_hi, g_hi - g_lo, g_cover, g_abs,
                 g_b_rmse, g_b_corr, g_b_max, g_b_mae, g_b_cover,
                 g.elapsed, missing, missing])
    push!(rows, ["IRT", N, K, "θ", θ_true, rep, "VB",
                 vb.m_θ[1], v_sd, v_lo, v_hi, v_hi - v_lo, v_cover, v_abs,
                 v_b_rmse, v_b_corr, v_b_max, v_b_mae, v_b_cover,
                 vb.elapsed, vb.n_iter, Int(vb.converged)])
    return rows
end

function run_testlet(N, K_per, T, θ_true_vec, rep, niter, burnin, master_seed)
    K = K_per * T
    testlet_of = repeat(1:T, inner=K_per)
    rng_data = MersenneTwister(hash((:tlt, N, K, T, rep, master_seed)))
    Y, info = BCSM.simulate_testlet_bcsm(rng_data, N, K;
                                          testlet_of=testlet_of,
                                          θ_true=θ_true_vec, σ_b=1.0)
    model = BCSM.TestletBCSM(K=K, testlet_of=testlet_of)

    rng_g = MersenneTwister(hash((:tlt_gibbs, N, K, T, rep, master_seed)))
    g = BCSM.gibbs_testlet_bcsm(Y, model; niter=niter, burnin=burnin, rng=rng_g)
    vb = BCSM.cavi_testlet_bcsm(Y, model; maxiter=400, tol=1e-7)

    g_b_post = vec(mean(g.samples_b, dims=1))
    g_b_rmse = sqrt(mean((g_b_post .- info.b) .^ 2))
    g_b_corr = cor(g_b_post, info.b)
    g_b_max  = maximum(abs.(g_b_post .- info.b))
    g_b_mae  = mean(abs.(g_b_post .- info.b))
    g_b_cover = b_coverage(g.samples_b, info.b)
    v_b_rmse = sqrt(mean((vb.m_b .- info.b) .^ 2))
    v_b_corr = cor(vb.m_b, info.b)
    v_b_max  = maximum(abs.(vb.m_b .- info.b))
    v_b_mae  = mean(abs.(vb.m_b .- info.b))
    v_b_cover = b_coverage_vb(vb.m_b, vb.v_b, info.b)

    rows = Vector{Vector{Any}}()
    for t in 1:T
        ch = g.samples_θ[:, t]
        gm, gs = mean(ch), std(ch)
        glo, ghi = quantile(ch, (0.025, 0.975))
        gcov = (glo ≤ θ_true_vec[t] ≤ ghi) ? 1 : 0
        gabs = abs(gm - θ_true_vec[t])
        push!(rows, ["TLT", N, K, "θ$t", θ_true_vec[t], rep, "MCMC",
                     gm, gs, glo, ghi, ghi - glo, gcov, gabs,
                     g_b_rmse, g_b_corr, g_b_max, g_b_mae, g_b_cover,
                     g.elapsed, missing, missing])

        vsd = sqrt(max(vb.v_θ[t], 0))
        vlo, vhi = vb.m_θ[t] - 1.96 * vsd, vb.m_θ[t] + 1.96 * vsd
        vcov = (vlo ≤ θ_true_vec[t] ≤ vhi) ? 1 : 0
        vabs = abs(vb.m_θ[t] - θ_true_vec[t])
        push!(rows, ["TLT", N, K, "θ$t", θ_true_vec[t], rep, "VB",
                     vb.m_θ[t], vsd, vlo, vhi, vhi - vlo, vcov, vabs,
                     v_b_rmse, v_b_corr, v_b_max, v_b_mae, v_b_cover,
                     vb.elapsed, vb.n_iter, Int(vb.converged)])
    end
    return rows
end

# ─────────────────────────── main ─────────────────────────── #

function main(args)
    opts = parse_args(args)
    d = design(opts.grid, opts.reps)
    out_path = something(opts.out,
        "results/param_recovery_$(opts.grid)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv")
    mkpath(dirname(out_path))

    @info "Parameter recovery grid" grid=opts.grid N=d.N_grid K=d.K_grid θ=d.θ_grid reps=d.reps

    cells = Tuple[]
    for N in d.N_grid, K in d.K_grid, θ_true in d.θ_grid
        push!(cells, (N, K, θ_true))
    end

    # Testlet cells: derive from grid by partitioning K into testlets when K ≥ 12
    testlet_cells = Tuple[]
    for N in d.N_grid, K in d.K_grid
        if K ≥ 12 && K % 4 == 0
            push!(testlet_cells, (N, K ÷ 4, 4, [0.25, 0.4, 0.3, 0.15]))
        elseif K ≥ 12 && K % 3 == 0
            push!(testlet_cells, (N, K ÷ 3, 3, [0.25, 0.4, 0.3]))
        end
    end

    all_rows = Vector{Vector{Any}}()
    t0 = time()

    @info "IRT-BCSM cells = $(length(cells)), Testlet-BCSM cells = $(length(testlet_cells)), reps = $(d.reps)"

    for (ci, (N, K, θ_true)) in enumerate(cells)
        for rep in 1:d.reps
            for row in run_irt(N, K, θ_true, rep, opts.niter, opts.burnin, opts.seed)
                push!(all_rows, row)
            end
        end
        elapsed = round(time() - t0, digits=1)
        @printf("[%6.1fs] IRT cell %d/%d  N=%d K=%d θ=%.2f  done (reps=%d)\n",
                elapsed, ci, length(cells), N, K, θ_true, d.reps)
        flush(stdout)
    end

    for (ci, (N, K_per, T, θv)) in enumerate(testlet_cells)
        for rep in 1:d.reps
            for row in run_testlet(N, K_per, T, θv, rep, opts.niter, opts.burnin, opts.seed)
                push!(all_rows, row)
            end
        end
        elapsed = round(time() - t0, digits=1)
        @printf("[%6.1fs] TLT cell %d/%d  N=%d K=%d T=%d  done (reps=%d)\n",
                elapsed, ci, length(testlet_cells), N, K_per * T, T, d.reps)
        flush(stdout)
    end

    # Write CSV
    open(out_path, "w") do io
        println(io, join(HEADER, ","))
        for r in all_rows
            println(io, join((x === missing ? "" : string(x) for x in r), ","))
        end
    end
    @info "DONE" elapsed=round(time()-t0, digits=1) rows=length(all_rows) out=out_path
end

main(ARGS)
