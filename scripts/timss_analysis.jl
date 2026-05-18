# TIMSS 2019 Grade 8 mathematics — BCSM analysis.
#
# This script loads a single booklet from the TIMSS 2019 international
# database, restricts to one country, fits both the IRT-BCSM and the
# Testlet-BCSM (with content-domain testlets), and prints a comparison of
# Gibbs and CAVI posteriors.
#
# Prerequisites:
#   1. Download the TIMSS 2019 SPSS data from
#        https://timss2019.org/international-database/
#      into data/raw/timss2019/   (BSGxxxM7.sav, BSAxxxM7.sav per country).
#   2. Install ReadStatTables.jl in the project environment:
#        julia --project=. -e 'using Pkg; Pkg.add("ReadStatTables")'
#
# Usage:
#   julia --project=. scripts/timss_analysis.jl --country USA --booklet 1
#
# This script intentionally keeps the heavy data-handling in user-controlled
# code rather than hidden inside the package — the BCSM model functions are
# generic and operate on any N × K binary matrix.

using BCSModels
using Random, Statistics, LinearAlgebra, Printf

function parse_args(args)
    country = "USA"
    booklet = 1
    data_root = "data/raw/timss2019"
    i = 1
    while i ≤ length(args)
        if args[i] == "--country"; country = args[i+1]; i += 2
        elseif args[i] == "--booklet"; booklet = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--data";    data_root = args[i+1]; i += 2
        else error("unknown argument $(args[i])") end
    end
    return (country=country, booklet=booklet, data_root=data_root)
end

# Placeholder: real loader pulls BSA<ctry>M7.sav (item responses) and
# BSG<ctry>M7.sav (student questionnaire) via ReadStatTables.jl, joins on
# IDSTUD, restricts to items in the requested booklet, and binarises the
# polytomous items into correct / incorrect. The mapping booklet → item codes
# is documented in TIMSS 2019 User Guide Appendix B / Item Information Files.
# Until those files are downloaded, this function raises an informative error.
function load_booklet(data_root, country, booklet)
    isdir(data_root) || error("""
        TIMSS data not found at $data_root. Steps to obtain it:
          1. Visit https://timss2019.org/international-database/
          2. Accept the data licence and download the SPSS bundle.
          3. Unzip into $data_root so that BSA$(country)M7.sav and
             BSG$(country)M7.sav are visible.
        Then re-run this script.
    """)

    have_reader = Base.find_package("ReadStatTables") !== nothing
    have_reader || error("""
        Install ReadStatTables.jl first:
          julia --project=. -e 'using Pkg; Pkg.add("ReadStatTables")'
    """)
    error("""
        load_booklet is a stub — the booklet → item-code mapping is in TIMSS
        2019 IDB Appendix B. Fill in the booklet $booklet items for country
        $country in scripts/timss_analysis.jl::load_booklet before running.
    """)
end

function content_domain_testlets(item_codes)
    # Map item codes to content domains using the TIMSS 2019 item-name convention
    # (prefix M??_ encodes the content cluster). Returns a vector of integer
    # testlet ids of length K. This stub assumes each unique prefix is its own
    # testlet; refine using the official ItemInformationFile when available.
    prefixes = unique([split(c, "_")[1] for c in item_codes])
    code_to_id = Dict(p => i for (i, p) in enumerate(prefixes))
    return [code_to_id[split(c, "_")[1]] for c in item_codes]
end

function main(args)
    opts = parse_args(args)
    @info "TIMSS 2019 Grade-8 Math BCSM analysis" opts...

    Y, item_codes, student_meta = load_booklet(opts.data_root, opts.country, opts.booklet)
    N, K = size(Y)
    @info "loaded responses" N K

    # --- IRT-BCSM ---
    println("\n=== IRT-BCSM (single-layer) ===")
    irt = BCSModels.IRTBCSM(K=K)
    g_irt = BCSModels.gibbs_irt_bcsm(Y, irt; niter=2000, burnin=1000,
                                 rng=MersenneTwister(1))
    θ_post = vec(g_irt.samples_θ)
    @printf("Gibbs θ posterior: mean=%.3f  sd=%.3f  95%% CI=(%.3f, %.3f)\n",
            mean(θ_post), std(θ_post),
            quantile(θ_post, 0.025), quantile(θ_post, 0.975))
    vb_irt = BCSModels.cavi_irt_bcsm(Y, irt; maxiter=500, tol=1e-7)
    sd_θ = sqrt(max(vb_irt.v_θ[1], 0))
    @printf("CAVI  θ variational: mean=%.3f  sd=%.3f  95%% CI=(%.3f, %.3f)\n",
            vb_irt.m_θ[1], sd_θ,
            vb_irt.m_θ[1] - 1.96 * sd_θ, vb_irt.m_θ[1] + 1.96 * sd_θ)
    @printf("Wall times: Gibbs %.2fs, CAVI %.2fs (%.1fx speedup)\n",
            g_irt.elapsed, vb_irt.elapsed, g_irt.elapsed / max(vb_irt.elapsed, 1e-6))

    # --- Testlet-BCSM with content-domain testlets ---
    println("\n=== Testlet-BCSM (content-domain testlets) ===")
    testlet_of = content_domain_testlets(item_codes)
    T = maximum(testlet_of)
    @info "testlet structure" T sizes=[count(==(t), testlet_of) for t in 1:T]
    tlt = BCSModels.TestletBCSM(K=K, testlet_of=testlet_of)
    g_tlt = BCSModels.gibbs_testlet_bcsm(Y, tlt; niter=2000, burnin=1000,
                                    rng=MersenneTwister(2))
    for t in 1:T
        ch = g_tlt.samples_θ[:, t]
        @printf("  Testlet %d: Gibbs θ=%.3f ± %.3f  CI=(%.3f, %.3f)\n",
                t, mean(ch), std(ch), quantile(ch, 0.025), quantile(ch, 0.975))
    end
    vb_tlt = BCSModels.cavi_testlet_bcsm(Y, tlt; maxiter=500, tol=1e-7)
    for t in 1:T
        sd = sqrt(max(vb_tlt.v_θ[t], 0))
        @printf("  Testlet %d: CAVI  θ=%.3f ± %.3f  CI=(%.3f, %.3f)\n",
                t, vb_tlt.m_θ[t], sd, vb_tlt.m_θ[t] - 1.96 * sd,
                vb_tlt.m_θ[t] + 1.96 * sd)
    end
    @printf("Wall times: Gibbs %.2fs, CAVI %.2fs (%.1fx speedup)\n",
            g_tlt.elapsed, vb_tlt.elapsed, g_tlt.elapsed / max(vb_tlt.elapsed, 1e-6))
end

main(ARGS)
