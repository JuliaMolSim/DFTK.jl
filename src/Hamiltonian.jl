using LinearAlgebra

# Data structure and functionality for a one-particle Hamiltonian
mutable struct Hamiltonian
    basis::PlaneWaveBasis  # Discretized model
    density                # The electron density used to build this Hamiltonian
    kinetic                # Discretized kinetic operator
    pot_external           # External local potential (e.g. Pseudopotential)
    pot_hartree            # Hartree potential
    pot_xc                 # XC potential
    pot_local              # Sum of all local potential
    pot_nonlocal           # Discretized non-local operator, e.g. non-local PSP projectors
end


# TODO Think about a better mechanism here
import Base: eltype
eltype(ham::Hamiltonian) = Complex{eltype(ham.basis.kpoints[1].coordinate)}

"""
Initialise a one-particle Hamiltonian from a model and optionally a density.
"""
Hamiltonian(basis::PlaneWaveBasis) = Hamiltonian(basis, RealFourierArray(basis))
function Hamiltonian(basis::PlaneWaveBasis{T}, ρ::RealFourierArray) where T
    model = basis.model
    potarray = similar(ρ.real, Complex{T})

    _, pot_external = model.build_external(basis, nothing, copy(potarray))
    _, pot_nonlocal = model.build_nonlocal(basis, nothing, copy(potarray))
    _, pot_hartree = model.build_hartree(basis, nothing, copy(potarray); ρ=ρ)
    _, pot_xc = model.build_xc(basis, nothing, copy(potarray); ρ=ρ)
    terms_local = filter(!isnothing, [pot_external, pot_hartree, pot_xc])
    pot_local = isempty(terms_local) ? nothing : .+(terms_local...)
    out = Hamiltonian(basis, ρ, Kinetic(basis), pot_external,
                      pot_hartree, pot_xc, pot_local, pot_nonlocal)
end

"""
Build / update an Hamiltonian out-of-place
"""
function update_hamiltonian(ham::Hamiltonian, ρ::RealFourierArray)
    nsimilar(::Nothing) = nothing
    nsimilar(x) = similar(x)
    ham = Hamiltonian(ham.basis, ρ, ham.kinetic, ham.pot_external, nsimilar(ham.pot_hartree),
                      nsimilar(ham.pot_xc), nsimilar(ham.pot_local), ham.pot_nonlocal)
    update_hamiltonian!(ham, ρ)
end

"""
Update Hamiltonian in-place
"""
function update_hamiltonian!(ham::Hamiltonian, ρ::RealFourierArray)
    basis = ham.basis
    model = basis.model
    ham.density = ρ
    model.build_hartree(basis, nothing, ham.pot_hartree; ρ=ρ)
    model.build_xc(basis, nothing, ham.pot_xc; ρ=ρ)
    if ham.pot_local !== nothing
        terms_local = filter(!isnothing, [ham.pot_external, ham.pot_hartree, ham.pot_xc])
        @assert !isempty(terms_local)
        ham.pot_local .= .+(terms_local...)
    end
    ham
end

"""
Compute and return electronic energies
"""
function update_energies!(energies, ham::Hamiltonian, Psi, occupation, ρ=nothing)
    # TODO Not too happy with this way of computing the energy
    #      ... a lot of Fourier transforms on the ρ

    basis = ham.basis
    model = basis.model
    model.spin_polarisation in (:none, :spinless) || error("$(model.spin_polarisation) not implemented")
    ρ === nothing && (ρ = compute_density(ham.basis, Psi, occupation))

    energies[:Kinetic] = energy_term_operator(ham.kinetic, Psi, occupation)

    function insert_energy!(key, builder; kwargs...)
        energy, _ = builder(basis, Ref{valtype(energies)}(0), nothing; kwargs...)
        energy !== nothing && (energies[key] = energy[])
    end
    insert_energy!(:PotExternal, model.build_external; ρ=ρ)
    insert_energy!(:PotHartree, model.build_hartree; ρ=ρ)
    insert_energy!(:PotXC, model.build_xc; ρ=ρ)
    insert_energy!(:PotNonLocal, model.build_nonlocal; Psi=Psi, occupation=occupation)

    energies
end
update_energies(ham::Hamiltonian, Psi, occupation, ρ=nothing) =
    update_energies!(Dict{Symbol, real(eltype(ham))}(), ham, Psi, occupation, ρ)


"""
Update energies and Hamiltonian, return energies
"""
function update_energies_hamiltonian!(energies, ham::Hamiltonian, Psi, occupation, ρ=nothing)
    # TODO this can be improved and made more efficient, by doing density and hamiltonian
    #      update and energy computation at the same time.
    ρ === nothing && (ρ = compute_density(ham.basis, Psi, occupation))
    update_hamiltonian!(ham, ρ)
    compute_energies!(energies, ham, Psi, occupation, ham.density)
end
