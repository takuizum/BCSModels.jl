# Documenter.jl entry point. Builds the BCSModels.jl manual locally with
#   julia --project=docs docs/make.jl
# CI runs the same command and deploys to GitHub Pages from .github/workflows/Documenter.yml.

using BCSModels
using Documenter
using Documenter: Remotes

DocMeta.setdocmeta!(BCSModels, :DocTestSetup, :(using BCSModels); recursive=true)

makedocs(;
    modules  = [BCSModels],
    authors  = "Takumi Itamiya <sep10.taku.izum@gmail.com>",
    sitename = "BCSModels.jl",
    repo     = Remotes.GitHub("takuizum", "BCSModels.jl"),
    format   = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://takuizum.github.io/BCSModels.jl",
        edit_link  = "main",
        assets     = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Theory" => [
            "Full derivation" => "theory.md",
            "Condensed implementation note" => "derivations.md",
            "Paper-to-code reading guide" => "reading_guide.md",
        ],
        "Experiments" => [
            "MCMC vs Variational Bayes" => "mcmc_vs_vb.md",
            "Parameter recovery" => "param_recovery.md",
        ],
        "API reference" => "api.md",
    ],
    # Doctests in docstrings are enabled by default. Use `Documenter.doctest`
    # in CI to detect drift between docstring examples and actual output.
    doctest = true,
    warnonly = [:missing_docs, :cross_references],  # don't fail CI on these
)

deploydocs(;
    repo   = "github.com/takuizum/BCSModels.jl",
    devbranch = "main",
    push_preview = true,
)
