# JET.jl static analysis for BCSModels.jl
#
# We use the "basic" static analysis (`report_package`) which catches
# undefined names, illegal calls, and other type-stability red flags
# without enforcing full type-stability (which is heavy on a probabilistic
# inference package).
#
# Failures here typically indicate a latent bug worth fixing before a
# registry release.

using JET
using BCSModels
using Test

@testset "JET.jl static analysis" begin
    # report_package is the recommended entry point for package-wide checks.
    # Treat any reported error or higher-severity finding as a test failure;
    # warnings (e.g. about untyped fields in user-facing structs) are
    # informational only.
    report = JET.report_package(BCSModels;
                                target_modules = (BCSModels,))
    n_reports = length(JET.get_reports(report))
    if n_reports > 0
        @info "JET reports — review at next maintenance pass" report
    end
    @test n_reports == 0
end
