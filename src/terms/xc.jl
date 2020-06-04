using Libxc
include("../xc/xc_evaluate.jl")

"""
Exchange-correlation term, defined by a list of functionals and usually evaluated through libxc.
"""
struct Xc
    functionals::Vector{Functional}
    scaling_factor::Real  # to scale by an arbitrary factor (useful for exploration)
end
Xc(functionals::Vector; scaling_factor=1) = Xc(functionals, scaling_factor)
Xc(functionals::Vector{Symbol}; kwargs...) = Xc(Functional.(functionals); kwargs...)
Xc(functional::Symbol; kwargs...) = Xc([Functional(functional)]; kwargs...)
Xc(functionals::Symbol...; kwargs...) = Xc([functionals...]; kwargs...)
(xc::Xc)(basis) = TermXc(basis, xc.functionals, xc.scaling_factor)

struct TermXc <: Term
    basis::PlaneWaveBasis
    functionals::Vector{Functional}
    scaling_factor::Real
end

function ene_ops(term::TermXc, ψ, occ; ρ, kwargs...)
    basis = term.basis
    T = eltype(basis)
    model = basis.model
    @assert all(xc.family in (:lda, :gga) for xc in term.functionals)

    # Take derivatives of the density if needed.
    max_ρ_derivs = maximum(max_required_derivative, term.functionals)
    density = DensityDerivatives(basis, max_ρ_derivs, ρ)

    potential = zeros(T, basis.fft_size)
    zk = zeros(T, basis.fft_size)  # Energy per unit particle
    E = zero(T)
    for xc in term.functionals
        # Evaluate the functional and its first derivative (potential)
        res = evaluate(xc; input_kwargs(xc.family, density)..., zk=zk, derivatives=1)

        # Add energy contribution
        dVol = basis.model.unit_cell_volume / prod(basis.fft_size)
        E += sum(res.zk .* ρ.real) * dVol

        # Add potential contribution from ∂(ρ E)/∂ρ and ∂(ρ E)/∂σ
        potential .+= res.vrho
        haskey(res, :vsigma) && (potential .+= potential_sigma(density, res.vsigma))
    end
    if term.scaling_factor != 1
        E *= term.scaling_factor
        potential .*= term.scaling_factor
    end

    ops = [RealSpaceMultiplication(basis, kpoint, potential) for kpoint in basis.kpoints]
    (E=E, ops=ops)
end


function compute_kernel(term::TermXc; kwargs...)
    error("Not implemented")
end

function apply_kernel(term::TermXc, dρ; ρ=ρ, kwargs...)
    basis = term.basis
    T = eltype(basis)
    @assert all(xc.family == :lda for xc in term.functionals)

    # Take derivatives of the density if needed.
    max_ρ_derivs = maximum(max_required_derivative, term.functionals)
    density = DensityDerivatives(basis, max_ρ_derivs, ρ)

    kernel = zeros(T, basis.fft_size)
    for xc in term.functionals
        # TODO This will also evaluate energy and potential ... should be optimised
        # in the next Libxc.jl one can just request derivatives=2:2
        res = evaluate(xc; input_kwargs(xc.family, density)..., derivatives=2)
        kernel += res.v2rho2
    end


    from_real(term.basis, kernel .* dρ.real)
end


#=
The total energy is
Etot = ∫ ρ E(ρ,σ), where σ = |∇ρ|^2 and E(ρ,σ) is the energy per unit particle
libxc provides the scalars
Vρ = ∂(ρ E)/∂ρ
Vσ = ∂(ρ E)/∂σ

Consider a variation dϕi of an orbital (considered real for
simplicity), and let dEtot be the corresponding variation of the
energy. Then the potential Vxc is defined by
dEtot = 2 ∫ Vxc ϕi dϕi

dρ = 2 ϕi dϕi
dσ = 2 ∇ρ ⋅ ∇dρ = 4 ∇ρ ⋅ ∇(ϕi dϕi)
dEtot = ∫ Vρ dρ + Vσ dσ
      = 2 ∫ Vρ ϕi dϕi + 4 ∫ Vσ ∇ρ ⋅ ∇(ϕi dϕi)
      = 2 ∫ Vρ ϕi dϕi - 4 ∫ div(Vσ ∇ρ) ϕi dϕi
where we performed an integration by parts in the last equation (boundary terms drop by periodicity).

Therefore,
Vxc = Vρ - 2 div(Vσ ∇ρ)

See eg Richard Martin, Electronic stucture, p. 158

A more algebraic way of deriving these for GGA, which helps see why
this is correct at the discrete level, is the following. Let

B the (Fourier, spherical) space of orbitals
C the (Fourier, cube) space of Fourier densities/potentials
C* the (real-space, cube) space of real-space densities/potentials

X : B -> C be the extension operator
R : C -> B the restriction (adjoint of X)
IF : C -> C* be the IFFT
F : C* -> be the FFT, adjoint of IF
K : C -> C be the gradient operator (multiplication by iK, assuming dimension 1 to
simplify notations)

Then

ρ = IF(Xϕ).^2
ρf = F ρ
∇ρf = K ρf
∇ρ = IF ∇ρf

dρ = 2 IF(Xϕ) .* IF(X dϕ)
d∇ρ = IF K F dρ
    = 2 IF K F (IF(Xϕ) .* IF(X dϕ))

Etot = sum(ρ . * E(ρ,∇ρ.^2))
dEtot = sum(Vρ .* dρ + 2 (Vσ .* ∇ρ) .* d∇ρ)
      = sum(Vρ .* dρ + (IF K F) (Vσ .* ∇ρ) .* dρ)
(IF K F is self-adjoint; this is the discrete integration by parts)

Now note that

sum(V .* dρ) = sum(V .* IF(Xϕ) .* IF(X dϕ))
             = sum(R(F(V .* IF(Xϕ))) .* dϕ)
and the result follows.
=#

function max_required_derivative(functional)
    functional.family == :lda && return 0
    functional.family == :gga && return 1
    return -1
end

struct DensityDerivatives
    basis
    max_derivative::Int
    ρ         # density on real-space grid
    ∇ρ_real   # density gradient on real-space grid
    σ_real    # contracted density gradient on real-space grid
end

"""
DOCME compute density in real space and its derivatives starting from Fourier-space density ρ
"""
function DensityDerivatives(basis, max_derivative::Integer, ρ)
    model = basis.model
    @assert model.spin_polarization == :none "Only spin_polarization == :none implemented."
    function ifft(x)
        tmp = G_to_r(basis, x)
        check_real(tmp)
        real(tmp)
    end

    ρF = ρ.fourier
    σ_real = nothing
    ∇ρ_real = nothing
    if max_derivative < 0 || max_derivative > 1
        error("max_derivative not in [0, 1]")
    elseif max_derivative > 0
        ∇ρ_real = [ifft(im * [(model.recip_lattice * G)[α] for G in G_vectors(basis)] .* ρF)
                   for α in 1:3]
        # TODO The above assumes CPU arrays
        σ_real = sum(∇ρ_real[α] .* ∇ρ_real[α] for α in 1:3)
    end

    DensityDerivatives(basis, max_derivative, ρ.real, ∇ρ_real, σ_real)
end

function input_kwargs(family, density)
    family == :lda && return (rho=density.ρ, )
    family == :gga && return (rho=density.ρ, sigma=density.σ_real)
    return NamedTuple()
end

# Evaluates the potential contribution from the gradient term -2 div(Vσ ∇ρ)
function potential_sigma(density, Vσ)
    basis = density.basis

    gradterm = sum(1:3) do α
        # Compute term inside -∇(  ) in Fourier space
        Vσ∇ρ = r_to_G(basis, complex(Vσ .* density.∇ρ_real[α]))

        # take derivative
        im * [(basis.model.recip_lattice * G)[α] for G in G_vectors(basis)] .* Vσ∇ρ
    end

    -2real(G_to_r(basis, gradterm))
end
