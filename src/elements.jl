using AtomsBase

# Data structure for chemical element and the potential model via which
# they interact with electrons. A compensating charge background is
# always assumed. It is assumed that each implementing struct
# defines at least the functions `local_potential_fourier` and `local_potential_real`.
# Very likely `species`, and `charge_ionic` need to be defined as well.
abstract type Element end

"""Return the chemical species corresponding to an element"""
AtomsBase.species(::Element) = ChemicalSpecies(0)  # dummy atom

"""Chemical symbol corresponding to an element"""
AtomsBase.element_symbol(el::Element) = element_symbol(species(el))

"""Return the atomic mass of an element"""
AtomsBase.mass(el::Element) = mass(species(el))

"""Return the total nuclear charge of an element"""
charge_nuclear(el::Element) = atomic_number(species(el))

"""Return the total ionic charge of an element (nuclear charge - core electrons)"""
charge_ionic(el::Element) = charge_nuclear(el)

"""Return the number of valence electrons"""
n_elec_valence(el::Element) = charge_ionic(el)

"""Return the number of core electrons"""
n_elec_core(el::Element) = charge_nuclear(el) - charge_ionic(el)

"""Check presence of model core charge density (non-linear core correction)."""
has_core_density(::Element) = false
# The preceding functions are fallback implementations that should be altered as needed.

# Fall back to the Gaussian table for Elements without pseudopotentials
function valence_charge_density_fourier(el::Element, p::T)::T where {T <: Real}
    gaussian_valence_charge_density_fourier(el, p)
end

"""Gaussian valence charge density using Abinit's coefficient table, in Fourier space."""
function gaussian_valence_charge_density_fourier(el::Element, p::T)::T where {T <: Real}
    charge_ionic(el) * exp(-(p * atom_decay_length(el))^2)
end

function core_charge_density_fourier(::Element, ::T)::T where {T <: Real}
    error("Abstract elements do not necesesarily provide core charge density.")
end

# Fallback print function:
Base.show(io::IO, el::Element) = print(io, "$(typeof(el))($(species(el)))")


#
# ElementCoulomb
#
struct ElementCoulomb <: Element
    species::ChemicalSpecies
    mass  # Atomic mass
end
AtomsBase.species(el::ElementCoulomb) = el.species
AtomsBase.mass(el::ElementCoulomb)    = el.mass
n_elec_core(el::ElementCoulomb) = 0

"""
Element interacting with electrons via a bare Coulomb potential
(for all-electron calculations)
`key` may be an element symbol (like `:Si`), an atomic number (e.g. `14`)
or a chemical species (e.g. `ChemicalSpecies(:He3)`)
"""
function ElementCoulomb(key::Union{Integer,Symbol,ChemicalSpecies};
                        mass=AtomsBase.mass(ChemicalSpecies(key)))
    ElementCoulomb(ChemicalSpecies(key), mass)
end

function local_potential_fourier(el::ElementCoulomb, p::T) where {T <: Real}
    p == 0 && return zero(T)  # Compensating charge background
    # General atom => Use default Coulomb potential
    # We use int_{R^3} -Z/r e^{-i p⋅x} = -4π Z / |p|^2
    Z = charge_nuclear(el)
    return -4T(π) * Z / p^2
end
local_potential_real(el::ElementCoulomb, r::Real) = -charge_nuclear(el) / r


#
# ElementPsp
#
struct ElementPsp{P} <: Element
    species::ChemicalSpecies
    psp::P  # Pseudopotential data structure
    mass    # Atomic mass
end
function Base.show(io::IO, el::ElementPsp)
    pspid = isempty(el.psp.identifier) ? "custom" : el.psp.identifier
    print(io, "ElementPsp($(el.species), \"$pspid\")")
end

"""
Element interacting with electrons via a pseudopotential model.
`key` may be an element symbol (like `:Si`), an atomic number (e.g. `14`)
or a chemical species (e.g. `ChemicalSpecies(:He3)`).
`psp` may be one of:
- a `PseudoPotentialData.PseudoFamily` to automatically determine the
  pseudopotential from the specified pseudo family. In this case
  `kwargs` are used when calling [`load_psp`](@ref) to read the pseudopotential
  from disk.
- a `Dict{Symbol,String}` mapping an atomic symbol to the pseudopotential
  to be employed.
  Again `kwargs` are used when calling [`load_psp`](@ref) to read the pseudopotential
  from disk.
- a pseudopotential object (like [`PspHgh`](@ref) or [`PspUpf`](@ref)),
  usually obtained using [`load_psp`](@ref)
- `nothing` (to return a `ElementCoulomb`)

## Examples
Construct an `ElementPsp` for silicon using a HGH pseudopotential from an identifier
```julia
ElementPsp(:Si, load_psp("psp/hgh/Si-q4"))
```
Construct an `ElementPsp` again for silicon using the specified pseudpotential family
(from the `PseudopotentialData` package).
```julia
using PseudoPotentialData
ElementPsp(:Si, PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf"))
```
"""
function ElementPsp(species::ChemicalSpecies, family::AbstractDict;
                    mass=AtomsBase.mass(species), kwargs...)
    psp = load_psp(family[element_symbol(species)]; kwargs...)
    ElementPsp(species, psp, mass)
end
function ElementPsp(species::ChemicalSpecies, psp; mass=AtomsBase.mass(species))
    ElementPsp(species, psp, mass)
end
function ElementPsp(species::ChemicalSpecies, psp::Nothing; kwargs...)
    ElementCoulomb(species; kwargs...)
end
function ElementPsp(key::Union{Integer,Symbol}, psp; kwargs...)
    ElementPsp(ChemicalSpecies(key), psp; kwargs...)
end
@deprecate ElementPsp(key; psp, kwargs...) ElementPsp(key, psp; kwargs...)

AtomsBase.mass(el::ElementPsp)    = el.mass
AtomsBase.species(el::ElementPsp) = el.species
charge_ionic(el::ElementPsp)      = charge_ionic(el.psp)
has_core_density(el::ElementPsp)  = has_core_density(el.psp)

function local_potential_fourier(el::ElementPsp, p::T) where {T <: Real}
    p == 0 && return zero(T)  # Compensating charge background
    eval_psp_local_fourier(el.psp, p)
end
local_potential_real(el::ElementPsp, r::Real) = eval_psp_local_real(el.psp, r)

function valence_charge_density_fourier(el::ElementPsp, p::T) where {T <: Real}
    if has_valence_density(el.psp)
        eval_psp_density_valence_fourier(el.psp, p)
    else
        gaussian_valence_charge_density_fourier(el, p)
    end
end
function core_charge_density_fourier(el::ElementPsp, p::T) where {T <: Real}
    eval_psp_density_core_fourier(el.psp, p)
end


#
# ElementCohenBergstresser
#
struct ElementCohenBergstresser <: Element
    species::ChemicalSpecies
    V_sym  # Map |G|^2 (in units of (2π / lattice_constant)^2) to form factors
    lattice_constant  # Lattice constant (in Bohr) which is assumed
end
AtomsBase.species(el::ElementCohenBergstresser) = el.species
charge_ionic(el::ElementCohenBergstresser) = 4

"""
Element where the interaction with electrons is modelled
as in [CohenBergstresser1966](https://doi.org/10.1103/PhysRev.141.789).
Only the homonuclear lattices of the diamond structure
are implemented (i.e. Si, Ge, Sn).

`key` may be an element symbol (like `:Si`), an atomic number (e.g. `14`)
or a chemical species (e.g. `ChemicalSpecies(:Si)`).
"""
function ElementCohenBergstresser(key::Union{Integer,Symbol,ChemicalSpecies};
                                  lattice_constant=nothing)
    species = ChemicalSpecies(key)

    # Form factors from Cohen-Bergstresser paper Table 2
    # Lattice constants from Table 1
    data = Dict(:Si => (; form_factors=Dict( 3 => -0.21u"Ry",
                                             8 =>  0.04u"Ry",
                                            11 =>  0.08u"Ry"),
                        lattice_constant=5.43u"Å"),
                :Ge => (; form_factors=Dict( 3 => -0.23u"Ry",
                                             8 =>  0.01u"Ry",
                                            11 =>  0.06u"Ry"),
                        lattice_constant=5.66u"Å"),
                :Sn => (; form_factors=Dict( 3 => -0.20u"Ry",
                                             8 =>  0.00u"Ry",
                                            11 =>  0.04u"Ry"),
                        lattice_constant=6.49u"Å"),
            )

    symbol = element_symbol(species)
    if !(symbol in keys(data))
        error("Cohen-Bergstresser potential not implemented for element $symbol.")
    end
    lattice_constant = austrip(@something lattice_constant data[symbol].lattice_constant)

    # Unit-cell volume of the primitive lattice (used in DFTK):
    unit_cell_volume = det(lattice_constant / 2 .* [[0 1 1]; [1 0 1]; [1 1 0]])

    # The form factors in the Cohen-Bergstresser paper Table 2 are
    # with respect to normalized planewaves (i.e. not plain Fourier coefficients)
    # and are already symmetrized into a sin-cos basis (see derivation p. 141)
    # => Scale by Ω / 2 to get them into the DFTK convention
    V_sym = Dict(key => austrip(value) * unit_cell_volume / 2
                 for (key, value) in pairs(data[symbol].form_factors))

    ElementCohenBergstresser(species, V_sym, lattice_constant)
end

function local_potential_fourier(el::ElementCohenBergstresser, p::T) where {T <: Real}
    p == 0 && return zero(T)  # Compensating charge background

    # Get |p|^2 in units of (2π / lattice_constant)^2
    psq_pi = Int(round(p^2 / (2π / el.lattice_constant)^2, digits=2))
    T(get(el.V_sym, psq_pi, 0.0))
end


#
# ElementGaussian
#
struct ElementGaussian{T} <: Element
    α::T  # Prefactor
    L::T  # Width of the Gaussian nucleus
    symbol::Symbol  # Element symbol
    mass  # Atomic mass
end
Base.show(io::IO, el::ElementGaussian) = print(io, "$(typeof(el))($(el.α), $(el.L))")

charge_nuclear(::ElementGaussian) = 0
AtomsBase.species(::ElementGaussian) = nothing
AtomsBase.mass(el::ElementGaussian) = el.mass
AtomsBase.element_symbol(el::ElementGaussian) = el.symbol

"""
Element interacting with electrons via a Gaussian potential.
Symbol is non-mandatory.
"""
function ElementGaussian(α, L; symbol=:X, mass=nothing)
    T = promote_type(typeof(α), typeof(L))
    ElementGaussian{T}(α, L, symbol, mass)
end
function local_potential_real(el::ElementGaussian, r)
    -el.α / (√(2π) * el.L) * exp(- (r / el.L)^2 / 2)
end
function local_potential_fourier(el::ElementGaussian, p::Real)
    -el.α * exp(- (p * el.L)^2 / 2)  # = ∫_ℝ³ V(x) exp(-ix⋅p) dx
end
