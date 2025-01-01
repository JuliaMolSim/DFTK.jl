module DFTKPseudoPotentialDataExt

using DFTK
using PseudoPotentialData

function DFTK.extra_psp_kwargs(family::PseudoFamily, element::Symbol)
    meta = pseudometa(family, element)
    haskey(meta, "rcut") ? (; rcut=meta["rcut"]) : (;)
end

end