"""
    BCSM

Bayesian Covariance Structure Modeling for Item Response Theory in Julia.

Implements two posterior-inference engines for the additive-covariance IRT
family of Fox, Mulder, Klotzke and colleagues:

* a closed-form Gibbs sampler (`gibbs_irt_bcsm`, `gibbs_testlet_bcsm`),
* an original mean-field coordinate-ascent variational Bayes routine
  (`cavi_irt_bcsm`, `cavi_testlet_bcsm`).

The two principal model types are [`IRTBCSM`](@ref) (single covariance
layer; Fox 2024) and [`TestletBCSM`](@ref) (one layer per testlet; Fox,
Wenzel & Klotzke 2021). See `docs/theory.md` for the full derivation.
"""
module BCSModels

using LinearAlgebra
using Random
using Statistics
using StatsBase
using StatsFuns
using LogExpFunctions
using SpecialFunctions
using Distributions

include("distributions/truncated_shifted_inverse_gamma.jl")
include("distributions/truncated_normal_moments.jl")
include("covariance/additive_covariance.jl")
include("models/irt_bcsm.jl")
include("models/testlet_bcsm.jl")
include("mcmc/latent_utility.jl")
include("mcmc/gibbs_irt.jl")
include("mcmc/gibbs_testlet.jl")
include("vb/cavi_irt.jl")
include("vb/cavi_testlet.jl")
include("data/simulation.jl")
include("diagnostics.jl")

export TruncatedShiftedInverseGamma,
       rand_tsig, mean_tsig, var_tsig,
       AdditiveCovariance, build_covariance, sherman_morrison_inv_diag,
       IRTBCSM, TestletBCSM,
       gibbs_irt_bcsm, gibbs_testlet_bcsm,
       cavi_irt_bcsm, cavi_testlet_bcsm,
       simulate_irt_bcsm, simulate_testlet_bcsm,
       posterior_summary

end # module
