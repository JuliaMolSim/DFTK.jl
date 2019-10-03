include("PlaneWaveModel.jl")
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

"""
Initialise a one-particle Hamiltonian from a model and a zero density.
"""
function Hamiltonian(basis::PlaneWaveModel, ρ)
    model = basis.model

    _, pot_external = model.build_external(basis, nothing, similar(ρ))
    _, pot_nonlocal = model.build_nonlocal(basis, nothing, similar(ρ))
    out = Hamiltonian(basis, ρ, Kinetic(basis), pot_external, similar(ρ), similar(ρ),
                      similar(ρ), pot_nonlocal)
    build_hamiltonian!(out, ρ)
end

"""
Build / update an Hamiltonian out-of-place
"""
build_hamiltonian(basis::PlaneWaveModel, ρ) = Hamiltonian(basis, ρ)
function build_hamiltonian(ham::Hamiltonian, ρ)
    build_hamiltonian!(Hamiltonian(basis, ρ, ham.kinetic, ham.pot_external,
                                   similar(ham.pot_hartree), similar(ham.pot_xc),
                                   similar(ham.pot_local), ham.pot_nonlocal), ρ)
end

"""
Update Hamiltonian in-place
"""
function build_hamiltonian!(ham::Hamiltonian, ρ)
    basis = ham.basis
    model = basis.model
    model.build_hartree(basis, nothing, ham.pot_hartree)
    model.build_xc(basis, nothing, ham.pot_xc)
    ham.pot_local .= sum_nothing(ham.pot_external, pot_hartree, pot_xc)
    ham
end


# TODO Maybe later have build_energy_hamiltonian
#      to build energies and Hamiltonian at the same time.
