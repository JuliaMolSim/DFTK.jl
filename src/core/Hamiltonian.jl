include("sum_nothing.jl")
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
    potarray(ρ) = similar(ρ, Complex{T})
    potarray(::Nothing) = zeros(Complex{T}, basis.fft_size)
    ρzero = something(ρ, zeros(T, basis.fft_size))

    _, pot_external = model.build_external(basis, nothing, potarray(ρ))
    _, pot_nonlocal = model.build_nonlocal(basis, nothing, potarray(ρ))
    _, pot_hartree = model.build_hartree(basis, nothing, potarray(ρ); ρ=ρzero)
    _, pot_xc = model.build_xc(basis, nothing, potarray(ρ); ρ=ρzero)
    out = Hamiltonian(basis, ρzero, Kinetic(basis), pot_external,
                      pot_hartree, pot_xc,
                      sum_nothing(pot_external, pot_hartree, pot_xc), pot_nonlocal)
end

"""
Build / update an Hamiltonian out-of-place
"""
function update_hamiltonian(ham::Hamiltonian, ρ)
    nsimilar(::Nothing) = nothing
    nsimilar(p) = similar(p)
    ham = Hamiltonian(basis, ρ, ham.kinetic, ham.pot_external, nsimilar(ham.pot_hartree),
                      nsimilar(ham.pot_xc), nsimilar(ham.pot_local), ham.pot_nonlocal)
    update_hamiltonian!(ham, ρ)
end

"""
Update Hamiltonian in-place
"""
function update_hamiltonian!(ham::Hamiltonian, ρ)
    basis = ham.basis
    model = basis.model
    model.build_hartree(basis, nothing, ham.pot_hartree; ρ=ρ)
    model.build_xc(basis, nothing, ham.pot_xc; ρ=ρ)
    if ham.pot_local !== nothing
        ham.pot_local .= sum_nothing(ham.pot_external, ham.pot_hartree, ham.pot_xc)
    end
    ham
end


# TODO Maybe later have build_energy_hamiltonian
#      to build energies and Hamiltonian at the same time.
