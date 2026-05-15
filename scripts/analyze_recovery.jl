# Aggregate the per-replicate output of scripts/param_recovery.jl.
#
# For each (model, N, K, θ_layer, θ_true) cell and each method (MCMC, VB),
# compute:
#   bias_θ, RMSE_θ, posterior SD, 95% coverage, mean CI width, abs error
#   bias_b (RMSE of b across items, averaged across reps)
#   b correlation
#   b 95% coverage rate (averaged)
#   wall time and speedup
# and print a Markdown table, plus a marginal table aggregated over θ_true
# but split by N×K so the recovery trend across sample size and test length
# is visible.
#
# Usage:
#   julia --project=. scripts/analyze_recovery.jl results/param_recovery_medium_*.csv

using Statistics, Printf

# ─────────────── CSV reader (handles "missing" empty strings) ─────────────── #
asf(s) = isempty(s) ? NaN : parse(Float64, s)

function load(path)
    lines = readlines(path)
    header = split(lines[1], ',')
    idx = Dict(name => i for (i, name) in enumerate(header))
    rows = [split(l, ',') for l in lines[2:end] if !isempty(strip(l))]
    return header, idx, rows
end

# ─────────────── aggregator ─────────────── #

# key columns identify a cell
const CELL_KEYS = ["model", "N", "K", "θ_layer", "θ_true"]
const STAT_COLS = ["θ_est", "θ_sd", "θ_ci_width", "θ_cover", "θ_abs_error",
                   "b_rmse", "b_corr", "b_max_abs_err", "b_mean_abs_err",
                   "b_cover_rate", "elapsed_s"]

function group_by_cell(idx, rows)
    groups = Dict{NTuple{5,String}, Dict{String, Vector{Vector{Float64}}}}()
    for row in rows
        key = (row[idx["model"]], row[idx["N"]], row[idx["K"]],
               row[idx["θ_layer"]], row[idx["θ_true"]])
        method = row[idx["method"]]
        stats = Float64[asf(row[idx[c]]) for c in STAT_COLS]
        gm = get!(groups, key, Dict{String, Vector{Vector{Float64}}}())
        push!(get!(gm, method, Vector{Vector{Float64}}()), stats)
    end
    return groups
end

# ─────────────── presentation ─────────────── #

# stat index helpers
const SIDX = Dict(c => i for (i, c) in enumerate(STAT_COLS))

function cell_summary(reps_matrix::Matrix{Float64}, truth::Float64)
    # reps_matrix: nreps × length(STAT_COLS)
    est       = reps_matrix[:, SIDX["θ_est"]]
    sd        = reps_matrix[:, SIDX["θ_sd"]]
    ci_w      = reps_matrix[:, SIDX["θ_ci_width"]]
    cover     = reps_matrix[:, SIDX["θ_cover"]]
    b_rmse    = reps_matrix[:, SIDX["b_rmse"]]
    b_corr    = reps_matrix[:, SIDX["b_corr"]]
    b_max     = reps_matrix[:, SIDX["b_max_abs_err"]]
    b_mae     = reps_matrix[:, SIDX["b_mean_abs_err"]]
    b_cov     = reps_matrix[:, SIDX["b_cover_rate"]]
    elapsed   = reps_matrix[:, SIDX["elapsed_s"]]
    return (
        bias_θ      = mean(est) - truth,
        rmse_θ      = sqrt(mean((est .- truth) .^ 2)),
        mean_sd     = mean(sd),
        mean_ciw    = mean(ci_w),
        coverage    = mean(cover),
        b_rmse_mean = mean(b_rmse),
        b_corr_mean = mean(b_corr),
        b_mae_mean  = mean(b_mae),
        b_cov_mean  = mean(b_cov),
        elapsed     = mean(elapsed),
    )
end

function print_per_cell(groups)
    cells = sort(collect(keys(groups)))
    println("\n## Per-cell parameter recovery\n")
    println("| model | N | K | θ_layer | truth | method | bias_θ | RMSE_θ | sd_θ | width_θ | cov_θ | b_RMSE | b_corr | b_MAE | b_cov | time (s) |")
    println("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key in cells
        truth = parse(Float64, key[5])
        for method in ("MCMC", "VB")
            haskey(groups[key], method) || continue
            mat = reduce(hcat, groups[key][method])'   # nreps × n_stats
            s = cell_summary(Matrix(mat), truth)
            @printf("| %s | %s | %s | %s | %.2f | %s | %+.3f | %.3f | %.3f | %.3f | %.2f | %.3f | %.3f | %.3f | %.2f | %.3f |\n",
                    key[1], key[2], key[3], key[4], truth, method,
                    s.bias_θ, s.rmse_θ, s.mean_sd, s.mean_ciw, s.coverage,
                    s.b_rmse_mean, s.b_corr_mean, s.b_mae_mean,
                    s.b_cov_mean, s.elapsed)
        end
    end
end

function print_NK_marginal(groups; model="IRT")
    # aggregate across θ_true within (model, N, K)
    margin = Dict{Tuple{String,String}, Dict{String, Vector{Float64}}}()
    for (key, gm) in groups
        key[1] == model || continue
        nk = (key[2], key[3])     # (N, K)
        truth = parse(Float64, key[5])
        for method in ("MCMC", "VB")
            haskey(gm, method) || continue
            mat = reduce(hcat, gm[method])'
            stats = cell_summary(Matrix(mat), truth)
            bag = get!(margin, nk, Dict{String, Vector{Float64}}())
            v = get!(bag, method, Float64[])
            append!(v, [stats.bias_θ, stats.rmse_θ, stats.coverage,
                        stats.b_rmse_mean, stats.b_corr_mean,
                        stats.b_cov_mean, stats.elapsed])
        end
    end
    # each method's vector has length 7 × n_θ_truth — reshape and average
    cells = sort(collect(keys(margin)), by=k -> (parse(Int, k[1]), parse(Int, k[2])))
    println("\n## $model marginal recovery (averaged across θ truth values)\n")
    println("| N | K | method | |bias_θ| | RMSE_θ | cov_θ | b_RMSE | b_corr | b_cov | time (s) |")
    println("|---|---|---|---|---|---|---|---|---|---|")
    for nk in cells
        for method in ("MCMC", "VB")
            haskey(margin[nk], method) || continue
            v = margin[nk][method]
            n_truth = length(v) ÷ 7
            mat = reshape(v, 7, n_truth)
            @printf("| %s | %s | %s | %.3f | %.3f | %.2f | %.3f | %.3f | %.2f | %.3f |\n",
                    nk[1], nk[2], method,
                    mean(abs.(mat[1, :])),
                    mean(mat[2, :]),
                    mean(mat[3, :]),
                    mean(mat[4, :]),
                    mean(mat[5, :]),
                    mean(mat[6, :]),
                    mean(mat[7, :]))
        end
    end
end

function print_overall(groups)
    bias_MC = Float64[]; bias_VB = Float64[]
    rmse_MC = Float64[]; rmse_VB = Float64[]
    cov_MC  = Float64[]; cov_VB  = Float64[]
    sd_MC   = Float64[]; sd_VB   = Float64[]
    bRMSE_MC = Float64[]; bRMSE_VB = Float64[]
    bcov_MC  = Float64[]; bcov_VB  = Float64[]
    speed    = Float64[]
    for (key, gm) in groups
        truth = parse(Float64, key[5])
        if !haskey(gm, "MCMC") || !haskey(gm, "VB"); continue; end
        mc = cell_summary(Matrix(reduce(hcat, gm["MCMC"])'), truth)
        vb = cell_summary(Matrix(reduce(hcat, gm["VB"])'),   truth)
        push!(bias_MC, mc.bias_θ); push!(bias_VB, vb.bias_θ)
        push!(rmse_MC, mc.rmse_θ); push!(rmse_VB, vb.rmse_θ)
        push!(cov_MC, mc.coverage); push!(cov_VB, vb.coverage)
        push!(sd_MC, mc.mean_sd); push!(sd_VB, vb.mean_sd)
        push!(bRMSE_MC, mc.b_rmse_mean); push!(bRMSE_VB, vb.b_rmse_mean)
        push!(bcov_MC, mc.b_cov_mean); push!(bcov_VB, vb.b_cov_mean)
        push!(speed, mc.elapsed / max(vb.elapsed, 1e-9))
    end
    sd_ratio = sd_VB ./ max.(sd_MC, 1e-9)
    println("\n## Overall (mean across cells)\n")
    @printf("|bias θ|: MCMC = %.3f    VB = %.3f\n",
            mean(abs.(bias_MC)), mean(abs.(bias_VB)))
    @printf("RMSE θ : MCMC = %.3f    VB = %.3f   (ratio VB/MCMC = %.2f)\n",
            mean(rmse_MC), mean(rmse_VB), mean(rmse_VB)/mean(rmse_MC))
    @printf("cov θ  : MCMC = %.2f      VB = %.2f\n",
            mean(cov_MC), mean(cov_VB))
    @printf("sd ratio (VB/MCMC) median = %.2f   mean = %.2f\n",
            median(sd_ratio), mean(sd_ratio))
    @printf("b RMSE : MCMC = %.3f    VB = %.3f\n",
            mean(bRMSE_MC), mean(bRMSE_VB))
    @printf("b cov  : MCMC = %.2f      VB = %.2f\n",
            mean(bcov_MC), mean(bcov_VB))
    @printf("speedup (MCMC/VB) median = %.1fx  mean = %.1fx\n",
            median(speed), mean(speed))
end

# ─────────────── main ─────────────── #

if length(ARGS) < 1
    error("usage: julia analyze_recovery.jl <path.csv>")
end
header, idx, rows = load(ARGS[1])
groups = group_by_cell(idx, rows)
print_per_cell(groups)
print_NK_marginal(groups; model="IRT")
if any(k -> k[1] == "TLT", keys(groups))
    print_NK_marginal(groups; model="TLT")
end
print_overall(groups)
