using DFTK
using Zygote
using ChainRulesCore
import FiniteDiff
using ForwardDiff
using LinearAlgebra

# NOTE: this snippet works on Zygote v0.6.17 but breaks on v0.6.18 (suspicion: due to `basis` argument in PlaneWaveBasis rrule)

struct CustomXC
    f # energy = ∫ f(ρ)
end
(custom_xc::CustomXC)(basis) = TermCustomXC(basis, custom_xc.f)

struct TermCustomXC <: DFTK.Term
    basis::PlaneWaveBasis
    f
end

function ChainRulesCore.rrule(T::Type{CustomXC}, f)
    return T(f), ΔTf -> (NoTangent(), ΔTf.f)
end

function DFTK.ene_ops(term::TermCustomXC, ψ, occ; ρ, kwargs...)
    @assert term.basis.model.spin_polarization ∈ (:none, :spinless)
    basis = term.basis
    T = eltype(basis)
    E = sum(term.f.(ρ)) * basis.dvol
    fp(ρx) = ForwardDiff.derivative(term.f, ρx)
    V = fp.(ρ)
    ops = [DFTK.RealSpaceMultiplication(basis, kpoint, V[:, :, :, kpoint.spin]) for kpoint in basis.kpoints]
    (E=E, ops=ops)
end

function DFTK.apply_kernel(term::TermCustomXC, δρ; ρ, kwargs...)
    # ρtot = total_density(ρ)
    # δρtot = total_density(δρ) 
    fp(ρx) = ForwardDiff.derivative(f, ρx)
    fpp(ρx) = ForwardDiff.derivative(fp, ρx)
    fpp.(fp.(ρ)) .* δρ
end


kgrid = [1, 1, 1]
Ecut = 5
H = ElementPsp(:H, psp=load_psp("hgh/lda/h-q1"))
positions = [
    [0.45312500031210007, 1/2, 1/2],
    [0.5468749996028622, 1/2, 1/2],
]
atoms = [H => positions]
lattice = 16 * Diagonal(ones(3))

function make_model(xc::CustomXC)
    terms = [Kinetic(), AtomicLocal(), xc] # TODO support more term types
    Model(lattice; atoms=atoms, terms=terms, temperature=1e-3)
end
make_basis(model::Model) = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
make_basis(xc::CustomXC) = make_basis(make_model(xc))

function total_energy(basis) # debug: do not use Hellman-Feynman
    scfres = self_consistent_field(basis, is_converged=DFTK.ScfConvergenceDensity(1e-4))
    ψ = scfres.ψ
    occupation = scfres.occupation
    # ρ = scfres.ρ
    ρ = compute_density(basis, ψ, occupation) # TODO reactivate without OOM
    energies = [DFTK.ene_ops(term, ψ, occupation; ρ=ρ).E for term in basis.terms]
    sum(energies)
end

make_f(param) = x -> param*sin(x)
xc = CustomXC(make_f(0.05))
basis = make_basis(xc)
scfres = self_consistent_field(basis, is_converged=DFTK.ScfConvergenceDensity(1e-4))

obj(param) = total_energy(make_basis(CustomXC(make_f(param))))
FiniteDiff.finite_difference_derivative(obj, 0.05) # 1.9947322350499912
# TODO 
# Zygote.gradient(obj, 0.05)

Zygote.gradient(xc -> DFTK.ene_ops(xc(basis), scfres.ψ, scfres.occupation; scfres.ρ).E, xc) # ((f = (param = 1.9947321925091828,),),) # correct

# Zygote.gradient(basis -> sum(compute_density(basis, scfres.ψ, scfres.occupation) .* scfres.ρ), basis) # out of memory

Zygote.gradient(obj, 0.05) # out of memory.. (look at compute_density TODO perhaps smaller setup)


DFTK._autodiff_compute_density(basis, scfres.ψ, scfres.occupation)
DFTK._autodiff_compute_density(basis, scfres.ψ, scfres.occupation) ≈ compute_density(basis, scfres.ψ, scfres.occupation) # true

@time DFTK._autodiff_compute_density(basis, scfres.ψ, scfres.occupation);

p = DFTK._compute_partial_density(basis, basis.kpoints[1], scfres.ψ[1], scfres.occupation[1])
p = real(p)
@time Zygote.gradient(
    basis -> real(sum(p .* DFTK._compute_partial_density(basis, basis.kpoints[1], scfres.ψ[1], scfres.occupation[1]))),
    basis
)

@time Zygote.gradient(basis -> sum(compute_density(basis, scfres.ψ, scfres.occupation)), basis)




# TODO rrule for self_consistent_field


# SCF fixed point map f(ρ; basis) = ρ
# function f(ρ; basis)
#   energies, ham = energy_hamiltonian(basis, ψ, occ; ρ)
#   nextstate = next_density(ham)
#   ψ, eigenvalues, occupation, εF, ρout = nextstate
#   ρout
# end
#
# function next_density(ham)
#   eigres = diagonalize(ham)
#   occupation, εF = compute_occupation(ham.basis, eigres.λ)
#   ρout = compute_density(ham.basis, eigres.X, occupation)
#   ρout
# end


# frule(basis, δbasis) sketch:
# function frule((δbasis,), ::typeof(self_consistent_field), basis; kwargs...)
#     scfres = self_consistent_field(basis; kwargs...)
#     ψ, occupation, ρ = scfres
#     (energies, H), (δenergies, δH) = frule((NoTangent(), δbasis, ZeroTangent(), ZeroTangent()), energy_hamiltonian, basis, ψ, occupation; ρ)
#     δHψ = δH * ψ
#     δψ = DFTK.solve_ΩplusK(basis, ψ, -δHψ, occupation)
#     δρ = DFTK.compute_δρ(basis, ψ, δψ, occupation)
#     δscfres = (ham=δH, basis=δbasis; ...ψ=δψ, ρ=δρ)
#     (scfres, δscfres)
# end

# (basis, δbasis) --o (scfres, δscfres)
# now try to transpose this as:
# (basis, ∂scfres) --o (scfres, ∂basis)
#


