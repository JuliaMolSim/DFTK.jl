"""
    recommended_cutoff(model::Model)

Return the recommended kinetic energy cutoff, supersampling and density cutoff for this model.
Values may be `missing` if the respective data cannot be determined. This may be because the
`PseudoFamily` of the peudopotentials is not known (`model.pseudofamily` is `nothing`)
or that there is no such tabulated data available for this `PseudoFamily`.
"""
function PseudoPotentialData.recommended_cutoff(model::Model)
    family = pseudofamily(model)
    function get_maximum(property, default=missing)
        isnothing(family) && return default
        result = maximum(model.atom_groups) do group
            atom = model.atoms[first(group)]
            getproperty(recommended_cutoff(family, element_symbol(atom)), property)
        end
        ismissing(result) ? default : result
    end

    Ecut = get_maximum(:Ecut)
    supersampling = get_maximum(:supersampling, 2.0)
    Ecut_density  = get_maximum(:Ecut_density, supersampling^2 * Ecut)

    (; Ecut, supersampling, Ecut_density)
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
