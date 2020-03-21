import IterTools

"""
    list_psp(element; functional, family, core, datadir_psp)

List the pseudopotential files known to DFTK. Allows various ways
to restrict the displayed files.

# Examples
```julia-repl
julia> list_psp(family="hgh")
```
will list all HGH-type pseudopotentials and
```julia-repl
julia> list_psp(family="hgh", functional="lda")
```
will only list those for LDA (also known as Pade in this context)
and
```julia-repl
julia> list_psp(:O, core=:semicore)
```
will list all oxygen semicore pseudopotentials known to DFTK.
"""
function list_psp(element=nothing; family=nothing, functional=nothing, core=nothing,
                  datadir_psp=datadir_psp())
    # Normalise input keys
    isnothing(element)    || (element = Symbol(PeriodicTable.elements[element].symbol))
    isnothing(functional) || (functional = lowercase(functional))
    isnothing(family)     || (family = lowercase(family))

    psplist = []
    for (root, _, files) in walkdir(datadir_psp)
        root = relpath(root, datadir_psp)
        for file in files
            base, ext = splitext(file)
            ext == ".sh" && continue                    # Ignore scripts
            count(isequal('-'), base) == 1 || continue  # Need exactly one '-' in filename
            count(isequal('/'), root) == 1 || continue  # family/functional

            f_family, f_functional = split(root, "/")
            f_element, f_nvalence = split(base, "-")
            f_nvalence[1] == 'q' || continue              # Need 'q' before valence number
            f_element = Symbol(uppercase(f_element[1]) * f_element[2:end])
            haskey(PeriodicTable.elements, Symbol(f_element)) || continue

            push!(psplist, (identifier=joinpath(root, file), family=f_family,
                  functional=f_functional, element=f_element,
                  n_elec_valence=parse(Int, f_nvalence[2:end]),
                  path=joinpath(datadir_psp, root, file)))
        end
    end

    # Annotate "core" property (:other, :semicore, :fullcore)
    per_elem = IterTools.groupby(psp -> (psp.family, psp.functional, psp.element), psplist)
    psp_per_element = map(per_elem) do elgroup
        @assert length(elgroup) > 0
        if length(elgroup) == 1
            cores = [(core=:fullcore, )]
        else
            cores = append!(fill((core=:other, ), length(elgroup) - 2),
                            [(core=:semicore, ), (core=:fullcore, )])
        end
        merge.(sort(elgroup, by=psp -> psp.n_elec_valence), cores)
    end

    filter(collect(Iterators.flatten(psp_per_element))) do psp
        !isnothing(element)    && psp.element    != element    && return false
        !isnothing(functional) && psp.functional != functional && return false
        !isnothing(family)     && psp.family     != family     && return false
        !isnothing(core)       && psp.core       != core       && return false
        true
    end
end
