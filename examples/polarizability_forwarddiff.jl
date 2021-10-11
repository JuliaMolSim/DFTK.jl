# # Polarizability by linear response using ForwardDiff

using DFTK
using LinearAlgebra
using ForwardDiff

a = 10.
lattice = a * I(3)  # cube of ``a`` bohrs
He = ElementPsp(:He, psp=load_psp("hgh/lda/He-q2"))
atoms = [He => [[1/2; 1/2; 1/2]]]  # Helium at the center of the box

kgrid = [1, 1, 1]  # no k-point sampling for an isolated system
Ecut = 30
tol = 1e-8

## dipole moment of a given density (assuming the current geometry)
function dipole(basis, ρ)
    rr = [a * (r[1] - 1/2) for r in r_vectors(basis)]
    d = sum(rr .* ρ) * basis.dvol
end

function dipole(ε)
    basis_ε = make_basis(ε)
    res_ε = self_consistent_field(basis_ε, tol=tol)
    dipole(basis_ε, res_ε.ρ)
end

function make_basis(ε, lattice)
    model_ε = model_DFT(lattice, atoms, [:lda_x, :lda_c_vwn];  
                        extra_terms=[ExternalFromReal(r -> -ε * (r[1] - a/2))],
                        symmetries=false)
    PlaneWaveBasis(model_ε; Ecut, kgrid)
end
make_basis(ε) = make_basis(ε, lattice)

# workaround: enforce the Model to promote its type from Float64 to Dual, by promoting the lattice too
make_basis(ε::T) where T <: ForwardDiff.Dual = make_basis(ε, T.(lattice))

# # Forward-mode implicit differentiation of SCF

# We keep both a non-dual basis, and a basis including Duals for easy bookkeeping (but redundant computation)
function self_consistent_field_dual(basis::PlaneWaveBasis, basis_dual::PlaneWaveBasis{T}; kwargs...) where T <: ForwardDiff.Dual
    scfres = self_consistent_field(basis; kwargs...)
    ψ = DFTK.select_occupied_orbitals(basis, scfres.ψ)
    filled_occ = DFTK.filled_occupation(basis.model)
    n_spin = basis.model.n_spin_components
    n_bands = div(basis.model.n_electrons, n_spin * filled_occ)
    # number of kpoints and occupation
    Nk = length(basis.kpoints)
    occupation = [filled_occ * ones(n_bands) for ik = 1:Nk]

    # promote everything eagerly to Dual numbers
    occupation_dual = [T.(occupation[1])]
    ψ_dual = [Complex.(T.(real(ψ[1])), T.(imag(ψ[1])))]
    ρ_dual = compute_density(basis_dual, ψ_dual, occupation_dual)

    _, δH = energy_hamiltonian(basis_dual, ψ_dual, occupation_dual; ρ=ρ_dual)
    δHψ = δH * ψ_dual
    δHψ = [ForwardDiff.partials.(δHψ[1], 1)]
    δψ = DFTK.solve_ΩplusK(basis, ψ, -δHψ, occupation)
    δρ = DFTK.compute_δρ(basis, ψ, δψ, occupation)
    ρ = ForwardDiff.value.(ρ_dual)
    ψ, ρ, δψ, δρ
end

function dipole(ε::ForwardDiff.Dual)
    T = ForwardDiff.tagtype(ε)
    basis = make_basis(ForwardDiff.value(ε))
    basis_dual = make_basis(ε)
    ψ, ρ, δψ, δρ = self_consistent_field_dual(basis, basis_dual; tol=tol)
    ρ_dual = ForwardDiff.Dual{T}.(ρ, δρ)
    dipole(basis_dual, ρ_dual)
end


polarizability = ForwardDiff.derivative(dipole, 0.0)
polarizability_ref = let 
    ε = 1e-4
    (dipole(ε) - dipole(0.0)) / ε
end

println("Polarizability via ForwardDiff: $polarizability")
println("Polarizability via finite difference: $polarizability_ref")
