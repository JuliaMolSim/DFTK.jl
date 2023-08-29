abstract type DensityConstructionMethod                  end
abstract type AtomicDensity <: DensityConstructionMethod end

struct RandomDensity           <: DensityConstructionMethod end
struct CoreDensity             <: AtomicDensity end
struct ValenceDensityGaussian  <: AtomicDensity end
struct ValenceDensityPseudo    <: AtomicDensity end
struct ValenceDensityAuto      <: AtomicDensity end

get_atomic_density(atom::Element, ::CoreDensity) = atom.potential.core_charge_density
function get_atomic_density(atom::Element, ::ValenceDensityPseudo)
    return atom.potential.ionic_charge_density
end
function get_atomic_density(atom::Element, ::ValenceDensityGaussian)
    # If the atom already has a Gaussian charge denstiy, return that
    if hasquantity(atom.potential, :ionic_charge_density)
        if typeof(atom.potential.ionic_charge_density) <: GaussianChargeDensity
            return atom.potential.ionic_charge_density
        end
    end
    # Otherwise, build a new one with default parameters
    return GaussianChargeDensity{RealSpace}(n_elec_core(atom), n_elec_valence(atom))
end
function get_atomic_density(atom::Element, ::ValenceDensityAuto)
    # If the atom already has an ionic charge density, return that
    if hasquantity(atom.potential, :ionic_charge_density)
        return atom.potential.ionic_charge_density
    end
    # Otherwise, build a default Gaussian charge density
    return get_atomic_density(atom, ValenceDensityGaussian())
end
function group_local_quantities(basis::PlaneWaveBasis, method::DensityConstructionMethod)
    model = basis.model
    atom_groups = model.atom_groups
    quantities = [get_atomic_density(model.atoms[first(group)], method)
                  for group in atom_groups]
    positions = [model.positions[group] for group in atom_groups]
    return (;quantities, positions)
end

function build_spin_density_superposition(
    basis::PlaneWaveBasis{T},
    densities::AbstractVector{<:AbstractQuantity},
    positions::Vector{Vector{Vec3{T}}},
    magnetic_moments=[]
) where {T}
    model = basis.model
    if model.spin_polarization in (:none, :spinless)
        isempty(magnetic_moments) && return nothing
        error("Initial magnetic moments can only be used with collinear models.")
    end

    # If no magnetic moments, start with a zero spin density
    magmoms = Vec3{T}[normalize_magnetic_moment(magmom) for magmom in magnetic_moments]
    if all(iszero, magmoms)
        @warn("Returning zero spin density guess, because no initial magnetization has " *
              "been specified in any of the given elements / atoms. Your SCF will likely " *
              "not converge to a spin-broken solution.")
        return zeros(T, basis.fft_size)
    end

    @assert length(magmoms) == length(basis.model.atoms)
    coefficients = map(basis.model.atoms, magmoms) do atom, magmom
        iszero(magmom[1:2]) || error("Non-collinear magnetization not yet implemented")
        magmom[3] ≤ n_elec_valence(atom) || error(
            "Magnetic moment $(magmom[3]) too large for element $(atomic_symbol(atom)) " *
            "with only $(n_elec_valence(atom)) valence electrons."
        )
        magmom[3] / n_elec_valence(atom)
    end::AbstractVector{T}  # Needed to ensure type stability in final guess density

    return build_atomic_superposition(basis, densities, positions; coefficients)
end

function guess_density(basis::PlaneWaveBasis{T},
                       method::DensityConstructionMethod=ValenceDensityAuto(),
                       magnetic_moments=[]; n_electrons=basis.model.n_electrons,
                       add_random=false) where {T}
    # Get the atomic densities and positions grouped by species
    atomic_densities, positions = group_local_quantities(basis, method)
    # Build the total charge density as a superposition of local atomic densities
    ρtot = build_atomic_superposition(basis, atomic_densities, positions)
    # Add random noise to break possibly spurious symmetries
    if add_random
        @static if VERSION < v"1.7"  # TaskLocalRNG not yet available.
            ρtot .+= rand(T, basis.fft_size)
        else
            ρtot .+= rand(TaskLocalRNG(), T, basis.fft_size)
        end
    end
    # Renormalize to the correct number of electrons
    N = sum(ρtot) * basis.dvol
    if !isnothing(n_electrons) && (N > 0)
        ρtot .*= n_electrons / N
    end
    # Build the spin charge density as a weighted superposition of local atomic densities
    ρspin = build_spin_density_superposition(basis, atomic_densities, positions,
                                             magnetic_moments)
    # Combine the total and spin densities -> ρ[Rx,Ry,Rz,(total,spin)]
    ρ_from_total_and_spin(ρtot, ρspin)
end

function guess_density(basis::PlaneWaveBasis, ::RandomDensity, args...;
                       n_electrons=basis.model.n_electrons, kwargs...)
    random_density(basis, n_electrons)
end

@doc raw"""
Build a random charge density normalized to the provided number of electrons.
"""
function random_density(basis::PlaneWaveBasis{T}, n_electrons::Integer) where {T}
    ρtot  = rand(T, basis.fft_size)
    ρtot  = ρtot .* n_electrons ./ (sum(ρtot) * basis.dvol)  # Integration to n_electrons
    ρspin = nothing
    if basis.model.n_spin_components > 1
        ρspin = rand((-1, 1), basis.fft_size) .* rand(T, basis.fft_size) .* ρtot
        @assert all(abs.(ρspin) .≤ ρtot)
    end
    ρ_from_total_and_spin(ρtot, ρspin)
end
