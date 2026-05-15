# Stub loader for TIMSS 2019 Grade 8 mathematics.
#
# The TIMSS 2019 international database is distributed as SPSS (.sav) and
# SAS (.sas7bdat) files at https://timss2019.org/international-database/. We
# read them with ReadStatTables.jl. To keep the package's hard dependency
# graph small, the loader checks for the optional dependency at runtime and
# emits an informative message if it is missing.
#
# The function returns:
#   - Y:           N × K binary response matrix (one of the dichotomous booklets)
#   - item_meta:   names, content domain, cognitive domain, testlet id
#   - student_meta: country, school, class, total weight
#
# The TIMSS booklet design means responses are sparse: each student sees only
# a subset of items. `load_timss_2019_booklet` therefore restricts to a single
# booklet's items to avoid the matrix-sampling missingness.

"""
    load_timss_2019_booklet(path; country=nothing, booklet=1) -> NamedTuple

Load the booklet `booklet` of TIMSS 2019 Grade 8 mathematics from the SPSS
data directory `path` (which should contain BSGxxxM7.sav and BSAxxxM7.sav for
countries `xxx`). Optionally restrict to a single country code.

This is a placeholder: the user must call this after running
`scripts/download_timss.sh`. The function will throw an informative error
until the user has downloaded the data and ReadStatTables.jl is installed.
"""
function load_timss_2019_booklet(path::AbstractString;
                                  country::Union{Nothing, AbstractString}=nothing,
                                  booklet::Int=1)
    isdir(path) || throw(ArgumentError(
        "TIMSS 2019 data directory not found: $path. Run scripts/download_timss.sh first."))
    has_pkg = Base.find_package("ReadStatTables") !== nothing
    has_pkg || error("""
        ReadStatTables.jl is required to load TIMSS .sav files.
        Install with:  using Pkg; Pkg.add("ReadStatTables")
        Then re-run this function.
    """)
    @info "TIMSS loader stub — implement booklet $booklet extraction in scripts/timss_analysis.jl"
    return (path=path, country=country, booklet=booklet,
            note="loader stub — extend in scripts/timss_analysis.jl after install")
end
