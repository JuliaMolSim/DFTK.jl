using LinearAlgebra

# Data structure and functionality for a one-particle Hamiltonian

struct Hamiltonian
    basis::PlaneWaveModel  # Discretized model
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
function Hamiltonian(basis::PlaneWaveModel{T}, ρ=nothing) where T
    model = basis.model
    # TODO This assumes CPU array
    potarray(p::Density) = similar(fourier(p))
    potarray(::Nothing) = zeros(Complex{T}, basis.fft_size)
    ρzero = something(ρ, density_zero(basis))

    _, pot_external = model.build_external(basis, nothing, potarray(ρ))
    _, pot_nonlocal = model.build_nonlocal(basis, nothing, potarray(ρ))
    _, pot_hartree = model.build_hartree(basis, nothing, potarray(ρ); ρ=ρzero)
    _, pot_xc = model.build_xc(basis, nothing, potarray(ρ); ρ=ρzero)
    pot_local = sum(term for term in (pot_external, pot_hartree, pot_xc)
                    if !isnothing(term))
    out = Hamiltonian(basis, ρzero, Kinetic(basis), pot_external,
                      pot_hartree, pot_xc, pot_local, pot_nonlocal)
end

"""
Build / update an Hamiltonian out-of-place
"""
function update_hamiltonian(ham::Hamiltonian, ρ::Density)
    nsimilar(::Nothing) = nothing
    nsimilar(p::Density) = similar(fourier(p))
    ham = Hamiltonian(ham.basis, ρ, ham.kinetic, ham.pot_external, nsimilar(ham.pot_hartree),
                      nsimilar(ham.pot_xc), nsimilar(ham.pot_local), ham.pot_nonlocal)
    update_hamiltonian!(ham, ρ)
end

"""
Update Hamiltonian in-place
"""
function update_hamiltonian!(ham::Hamiltonian, ρ::Density)
    basis = ham.basis
    model = basis.model
    model.build_hartree(basis, nothing, ham.pot_hartree; ρ=ρ)
    model.build_xc(basis, nothing, ham.pot_xc; ρ=ρ)
    if ham.pot_local !== nothing
        ham.pot_local .= sum(term for term in (ham.pot_external, ham.pot_hartree, ham.pot_xc)
                             if !isnothing(term))
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
    model.spin_polarisation == :none || error("$(model.spin_polarisation) not implemented")
    ρ === nothing && (ρ = compute_density(ham.basis, Psi, occupation))

    energies[:Kinetic] = energy_term_operator(ham.kinetic, Psi, occupation)
    if ham.pot_external !== nothing
        dVol = model.unit_cell_volume / prod(basis.fft_size)
        energies[:PotExternal] = real(sum(real(ρ) .* ham.pot_external) * dVol)
    end

    function insert_energy!(key, builder; kwargs...)
        energy, _ = builder(basis, Ref{valtype(energies)}(0), nothing; kwargs...)
        energy !== nothing && (energies[key] = energy[])
    end
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
