"""
    recommended_cutoff(model::Model)

Return the recommended kinetic energy cutoff, supersampling and density cutoff for this DFTK `model`.
Values may be `missing` if the respective data cannot be determined. This may be because the
`PseudoFamily` of the peudopotentials is not known (`model.pseudofamily` is `nothing`)
or that there is no such tabulated data available for this `PseudoFamily`.
"""
function PseudoPotentialData.recommended_cutoff(model::Model)
    function get_maximum(property)
        maximum(model.atom_groups) do group
            atom = model.atoms[first(group)]
            getproperty(recommended_cutoff(atom), property)
        end
    end
    (; Ecut          = get_maximum(:Ecut),
       supersampling = get_maximum(:supersampling),
       Ecut_density  = get_maximum(:Ecut_density))
end

"""
    recommended_cutoff(element::Element)

Return the recommended kinetic energy cutoff, supersampling and density cutoff for this `element`.
Values may be `missing` if the data cannot be determined.
"""
function PseudoPotentialData.recommended_cutoff(el::Element)
    data = (; Ecut=missing, supersampling=missing, Ecut_density=missing)
    function getdefault(data, key, default)
        ismissing(getproperty(data, key)) ? default : getproperty(data, key)
    end

    family = pseudofamily(el)
    if !isnothing(family)
        data = recommended_cutoff(family, element_symbol(el))
    end
    Ecut          = getdefault(data, :Ecut,          missing)
    supersampling = getdefault(data, :supersampling, 2.0)
    Ecut_density  = getdefault(data, :Ecut_density,  supersampling^2 * Ecut)

    (; Ecut, supersampling, Ecut_density)
end

"""
    pseudometa(element::Element)

Return the stored metadata for the pseudopotential definition used within
the DFTK element, if available.
Effectively this returns `pseudometa(pseudofamily(element), element_symbol(element))`.
"""
function PseudoPotentialData.pseudometa(el::Element)
    family = pseudofamily(el)
    if isnothing(family)
        return Dict{String,Any}()
    else
        return pseudometa(family, element_symbol(el))
    end
end

"""
    pseudofamily(model::Model)

Return the common family of pseudopotentials used in the Model, if a single
such family exists and can be determined from the `model.atoms`, else `nothing`.
"""
function pseudofamily(model::Model)
    has_common_family = allequal(model.atom_groups) do group
        pseudofamily(model.atoms[first(group)])
    end
    if has_common_family && !isempty(model.atoms)
        return pseudofamily(model.atoms[1])
    else
        return nothing
    end
end
