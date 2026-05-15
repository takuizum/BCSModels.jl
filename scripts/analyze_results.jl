# Aggregate the per-replicate output of scripts/sim_mcmc_vs_vb.jl into a per-cell
# summary that lets us check the claim:
#
#   "MCMC performs best, but VB is close. VB has some bias and tends to
#    underestimate the posterior SD, but is practically useful."
#
# We compute, per (model, N, K, param, truth):
#   bias_MCMC  = mean(est_MCMC  - truth)
#   bias_VB    = mean(est_VB    - truth)
#   rmse_MCMC  = sqrt(mean((est_MCMC  - truth)^2))
#   rmse_VB    = sqrt(mean((est_VB    - truth)^2))
#   posterior_sd_MCMC = mean(sd_MCMC)
#   posterior_sd_VB   = mean(sd_VB)
#   sd_ratio = posterior_sd_VB / posterior_sd_MCMC   (<1 ⇒ VB under-disperses)
#   ci_width_ratio = ci_VB / ci_MCMC
#   cover_MCMC, cover_VB = mean(cover_*)
#   speedup = mean(time_MCMC) / mean(time_VB)
#
# Usage:  julia --project=. scripts/analyze_results.jl results/sim_full_reps20.csv

using Statistics, Printf

function parse_csv(path)
    lines = readlines(path)
    header = split(lines[1], ',')
    rows = [split(l, ',') for l in lines[2:end] if !isempty(strip(l))]
    return header, rows
end

# parse helper
asf(s) = parse(Float64, s)
asi(s) = parse(Int, s)

function aggregate(path)
    header, rows = parse_csv(path)
    # build index
    idx = Dict(name => i for (i, name) in enumerate(header))
    # group by (model, N, K, param, truth) -> (method -> list of stats)
    groups = Dict{Tuple{String,String,String,String,String}, Dict{String, Vector{Vector{Float64}}}}()
    for row in rows
        key = (row[idx["model"]], row[idx["N"]], row[idx["K"]],
               row[idx["param"]], row[idx["truth"]])
        method = row[idx["method"]]
        elapsed_col = haskey(idx, "elapsed_s") ? idx["elapsed_s"] : idx["elapsed"]
        # est, sd, ci_lo, ci_hi, ci_width, cover, elapsed
        stats = [asf(row[idx["estimate"]]),
                 asf(row[idx["sd"]]),
                 asf(row[idx["ci_lo"]]),
                 asf(row[idx["ci_hi"]]),
                 asf(row[idx["ci_width"]]),
                 asf(row[idx["cover"]]),
                 asf(row[elapsed_col])]
        g = get!(groups, key, Dict{String, Vector{Vector{Float64}}}())
        push!(get!(g, method, Vector{Vector{Float64}}()), stats)
    end
    return groups
end

function summarise(groups)
    cells = sort(collect(keys(groups)))
    println("model | N | K | param | truth | bias_MC | bias_VB | rmse_MC | rmse_VB | sd_MC | sd_VB | sd_ratio | width_MC | width_VB | cov_MC | cov_VB | t_MC | t_VB | speedup")
    println("---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---")
    bias_VB_all = Float64[]; bias_MC_all = Float64[]
    sd_ratio_all = Float64[]
    cov_MC_all = Float64[]; cov_VB_all = Float64[]
    speedup_all = Float64[]
    rmse_VB_all = Float64[]; rmse_MC_all = Float64[]

    for key in cells
        truth = parse(Float64, key[5])
        g = groups[key]
        if !haskey(g, "MCMC") || !haskey(g, "VB"); continue; end
        mc = reduce(hcat, g["MCMC"])'   # nreps × 7
        vb = reduce(hcat, g["VB"])'
        nreps = size(mc, 1)
        est_mc, sd_mc, w_mc, cov_mc, t_mc = mc[:,1], mc[:,2], mc[:,5], mc[:,6], mc[:,7]
        est_vb, sd_vb, w_vb, cov_vb, t_vb = vb[:,1], vb[:,2], vb[:,5], vb[:,6], vb[:,7]
        bias_mc = mean(est_mc) - truth
        bias_vb = mean(est_vb) - truth
        rmse_mc = sqrt(mean((est_mc .- truth) .^ 2))
        rmse_vb = sqrt(mean((est_vb .- truth) .^ 2))
        sd_ratio = mean(sd_vb) / mean(sd_mc)
        speedup  = mean(t_mc) / mean(t_vb)
        @printf("%s | %s | %s | %s | %.2f | %+.3f | %+.3f | %.3f | %.3f | %.3f | %.3f | %.2f | %.3f | %.3f | %.2f | %.2f | %.3f | %.3f | %.1fx\n",
                key[1], key[2], key[3], key[4], truth,
                bias_mc, bias_vb, rmse_mc, rmse_vb,
                mean(sd_mc), mean(sd_vb), sd_ratio,
                mean(w_mc), mean(w_vb),
                mean(cov_mc), mean(cov_vb),
                mean(t_mc), mean(t_vb), speedup)

        push!(bias_VB_all, bias_vb); push!(bias_MC_all, bias_mc)
        push!(sd_ratio_all, sd_ratio)
        push!(cov_MC_all, mean(cov_mc)); push!(cov_VB_all, mean(cov_vb))
        push!(speedup_all, speedup)
        push!(rmse_VB_all, rmse_vb); push!(rmse_MC_all, rmse_mc)
    end

    println("\n## Overall (mean across cells)")
    @printf("bias    : MCMC = %+.3f   VB = %+.3f\n",
            mean(bias_MC_all), mean(bias_VB_all))
    @printf("|bias|  : MCMC = %.3f    VB = %.3f\n",
            mean(abs.(bias_MC_all)), mean(abs.(bias_VB_all)))
    @printf("RMSE    : MCMC = %.3f    VB = %.3f\n",
            mean(rmse_MC_all), mean(rmse_VB_all))
    @printf("sd ratio (VB / MCMC) median = %.3f   mean = %.3f\n",
            median(sd_ratio_all), mean(sd_ratio_all))
    @printf("coverage: MCMC = %.2f      VB = %.2f\n",
            mean(cov_MC_all), mean(cov_VB_all))
    @printf("speedup (MCMC time / VB time) median = %.1fx  mean = %.1fx\n",
            median(speedup_all), mean(speedup_all))
end

if length(ARGS) < 1
    error("usage: julia analyze_results.jl <csv>")
end
groups = aggregate(ARGS[1])
summarise(groups)
