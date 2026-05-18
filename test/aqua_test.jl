# Aqua.jl quality auditor for BCSModels.jl
#
# Aqua runs a battery of static checks that the Julia registry-quality
# community considers important:
#   - no ambiguous methods on the package's own types
#   - no unbound type parameters
#   - all dependencies listed in Project.toml are actually used
#   - no stale `[compat]` entries
#   - no piracy of types from base or other packages
#   - project metadata is well-formed
#
# This file is invoked from test/runtests.jl.

using Aqua
using BCSModels
using Test

@testset "Aqua.jl quality checks" begin
    Aqua.test_all(
        BCSModels;
        # We declare no stale deps because Project.toml was cleaned up to
        # contain only used packages. If you add a dependency, run
        #   Aqua.test_stale_deps(BCSModels)
        # locally and remove anything it reports.
        ambiguities = (; recursive = false),
    )
end
