using DFTK
using Zygote
using ChainRulesCore
import FiniteDiff
using ForwardDiff
using LinearAlgebra


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

function DFTK.ene_ops(term::TermCustomXC, basis, ψ, occ; ρ, kwargs...)
    @assert basis.model.spin_polarization ∈ (:none, :spinless)
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
lattice = 16 * Mat3(Diagonal(ones(3)))

function make_model(xc::CustomXC)
    terms = [Kinetic(), AtomicLocal(), xc] # TODO support more term types
    Model(lattice, atoms, terms)
end
make_basis(model::Model) = PlaneWaveBasis(model; Ecut, kgrid=kgrid)
make_basis(xc::CustomXC) = make_basis(make_model(xc))

function total_energy(basis) # debug: do not use Hellman-Feynman
    scfres = self_consistent_field(basis, is_converged=DFTK.ScfConvergenceDensity(1e-4))
    ψ = scfres.ψ
    occupation = scfres.occupation
    ρ = compute_density(basis, ψ, occupation)
    energies = [DFTK.ene_ops(term, basis, ψ, occupation; ρ=ρ).E for term in basis.terms]
    sum(energies)
end

make_f(param) = x -> param*sin(x)
xc = CustomXC(make_f(0.05))
basis = make_basis(xc)
scfres = self_consistent_field(basis, is_converged=DFTK.ScfConvergenceDensity(1e-4))

obj(param) = total_energy(make_basis(CustomXC(make_f(param))))
FiniteDiff.finite_difference_derivative(obj, 0.05) # 1.9947322350499912

Zygote.gradient(xc -> DFTK.ene_ops(xc(basis), scfres.ψ, scfres.occupation; scfres.ρ).E, xc) # ((f = (param = 1.9947321925091828,),),) # correct

Zygote.gradient(obj, 0.05) # (1.9947322203506723,)
