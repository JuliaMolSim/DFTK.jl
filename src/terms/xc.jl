using Libxc
include("../xc/xc_evaluate.jl")

"""
Exchange-correlation term, defined by a list of functionals and usually evaluated through libxc.
"""
struct Xc
    symbols::Vector{Symbol}  # Symbols of the functionals (uses Libxc.jl / libxc convention)
    scaling_factor::Real     # to scale by an arbitrary factor (useful for exploration)
end
Xc(symbols::Vector{Symbol}; scaling_factor=1) = Xc(symbols, scaling_factor)
Xc(symbols::Symbol...; kwargs...) = Xc([symbols...]; kwargs...)

function (xc::Xc)(basis)
    n_spin = length(spin_components(basis.model))
    functionals = Functional.(xc.symbols; n_spin=n_spin)
    TermXc(basis, functionals, xc.scaling_factor)
end

struct TermXc <: Term
    basis::PlaneWaveBasis
    functionals::Vector{Functional}
    scaling_factor::Real
end

function ene_ops(term::TermXc, ψ, occ; ρ, ρspin=nothing, kwargs...)
    basis = term.basis
    T = eltype(basis)
    model = basis.model
    @assert all(xc.family in (:lda, :gga) for xc in term.functionals)
    n_spin = length(spin_components(model))

    if isempty(term.functionals)
        ops = [NoopOperator(term.basis, kpoint) for kpoint in term.basis.kpoints]
        return (E=0, ops=ops)
    end

    # Take derivatives of the density if needed.
    max_ρ_derivs = maximum(max_required_derivative, term.functionals)
    density = DensityDerivatives(basis, max_ρ_derivs, ρ, ρspin)

    potential = [zeros(T, basis.fft_size) for _ in 1:n_spin]
    zk = zeros(T, basis.fft_size)  # Energy per unit particle
    E = zero(T)
    for xc in term.functionals
        # Evaluate the functional and its first derivative (potential)
        terms = evaluate(xc; input_kwargs(xc.family, density)..., zk=zk)

        # Add energy contribution
        dVol = basis.model.unit_cell_volume / prod(basis.fft_size)
        E += sum(terms.zk .* ρ.real) * dVol

        # Add potential contributions Vρ -2 ∇⋅(Vσ ∇ρ)
        for iσ in 1:n_spin
            potential[iσ] .+= @view terms.vrho[iσ, :, :, :]
        end
        if haskey(terms, :vsigma)
            @assert term.basis.model.spin_polarization in (:none, :spinless)
            potential[1] .+=
                -2 * divergence_real(
                    α -> @view(terms.vsigma[1, :, :, :]) .* density.∇ρ_real[α],
                    density.basis,
                )
        end
    end
    if term.scaling_factor != 1
        E *= term.scaling_factor
        potential = [pot .*= term.scaling_factor for pot in potential]
    end

    ops = [RealSpaceMultiplication(basis, kpoint, potential[index_spin(kpoint)])
           for kpoint in basis.kpoints]
    (E=E, ops=ops)
end


function compute_kernel(term::TermXc; ρ::RealFourierArray, kwargs...)
    @assert term.basis.model.spin_polarization in (:none, :spinless)
    kernel = similar(ρ.real)
    kernel .= 0
    for xc in term.functionals
        xc.family == :lda || error("compute_kernel only implemented for LDA")
        terms = evaluate(xc; rho=ρ.real, derivatives=2:2)  # only valid for LDA
        kernel .+= terms.v2rho2
    end
    Diagonal(vec(term.scaling_factor .* kernel))
end


function apply_kernel(term::TermXc, dρ::RealFourierArray; ρ::RealFourierArray, kwargs...)
    basis = term.basis
    T = eltype(basis)
    @assert all(xc.family in (:lda, :gga) for xc in term.functionals)
    isempty(term.functionals) && return nothing
    @assert basis.model.spin_polarization in (:none, :spinless)

    # Take derivatives of the density and the perturbation if needed.
    max_ρ_derivs = maximum(max_required_derivative, term.functionals)
    density = DensityDerivatives(basis, max_ρ_derivs, ρ)
    perturbation = DensityDerivatives(basis, max_ρ_derivs, dρ)

    result = similar(ρ.real)
    result .= 0
    for xc in term.functionals
        # TODO LDA actually only needs the 2nd derivatives for this ... could be optimised
        terms = evaluate(xc; input_kwargs(xc.family, density)..., derivatives=1:2)

        # Accumulate LDA and GGA terms in result
        result .+= terms.v2rho2 .* dρ.real
        if haskey(terms, :v2rhosigma)
            result .+= apply_kernel_term_gga(terms, density, perturbation)
        end
    end
    from_real(basis, term.scaling_factor .* result)
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
where we performed an integration by parts in the last equation
(boundary terms drop by periodicity).

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
    error("Functional family $(functional.family) not known.")
end

struct DensityDerivatives
    basis
    max_derivative::Int
    ρ         # density on real-space grid
    ∇ρ_real   # density gradient on real-space grid
    σ_real    # contracted density gradient on real-space grid
end

"""
DOCME compute density in real space and its derivatives starting from ρ
"""
function DensityDerivatives(basis, max_derivative::Integer, ρ::RealFourierArray, ρspin=nothing)
    model = basis.model
    @assert model.spin_polarization in (:collinear, :none, :spinless)
    @assert max_derivative in (0, 1)
    model.spin_polarization == :collinear && (@assert !isnothing(ρspin))
    ifft(x) = real_checked(G_to_r(basis, clear_without_conjugate!(x)))

    ρF = ρ.fourier
    σ_real = nothing
    ∇ρ_real = nothing
    if max_derivative > 0
        ∇ρ_real = [ifft(im * [G[α] for G in G_vectors_cart(basis)] .* ρF)
                   for α in 1:3]
        # TODO The above assumes CPU arrays
        σ_real = sum(∇ρ_real[α] .* ∇ρ_real[α] for α in 1:3)
    end
    if model.spin_polarization == :collinear
        @assert max_derivative == 0
        ρα = (ρ.real + ρspin.real) / 2
        ρβ = (ρ.real - ρspin.real) / 2

        # TODO This is the wrong place. This is specific to the interface towards Libxc.jl ...
        ρ_real = vcat(reshape(ρα, 1, basis.fft_size...), reshape(ρβ, 1, basis.fft_size...))
    else
        # TODO This is the wrong place. This is specific to the interface towards Libxc.jl ...
        ρ_real = reshape(ρ.real, 1, basis.fft_size...)
    end

    DensityDerivatives(basis, max_derivative, ρ_real, ∇ρ_real, σ_real)
end

function input_kwargs(family, density)
    family == :lda && return (rho=density.ρ, )
    family == :gga && return (rho=density.ρ, sigma=density.σ_real)
    return NamedTuple()
end

"""
Compute divergence of an operand function, which returns the cartesian x,y,z
components in real space when called with the arguments 1 to 3.
The divergence is also returned as a real-space array.
"""
function divergence_real(operand, basis)
    gradsum = sum(1:3) do α
        operand_α = r_to_G(basis, complex(operand(α)))
        del_α = im * [G[α] for G in G_vectors_cart(basis)]
        del_α .* operand_α
    end
    real(G_to_r(basis, gradsum))
end

function apply_kernel_term_gga(terms, density, perturbation)
    error("apply_kernel_term_gga is not yet working")
    # Follows DOI 10.1103/PhysRevLett.107.216402
    #
    # For GGA V = Vρ - 2 ∇⋅(Vσ ∇ρ) = (∂ε/∂ρ) - 2 ∇⋅((∂ε/∂σ) ∇ρ)
    #
    # dV(r) = f(r,r') dρ(r') = (∂V/∂ρ) dρ + (∂V/∂σ) dσ
    #
    # therefore
    # dV(r) = (∂^2ε/∂ρ^2) dρ - 2 ∇⋅((∂^2ε/∂σ∂ρ) ∇ρ) dρ
    #       + (∂^2ε/∂ρ∂σ) dσ - 2 ∇⋅((∂^ε/∂σ^2) ∇ρ + (∂ε/∂σ) (∂∇ρ/∂σ)) dσ
    #
    # Note dσ = 2∇ρ ∇dρ = 2∇ρ ∇dρ, therefore
    #    - 2 ∇⋅((∂ε/∂σ) (∂∇ρ/∂σ)) dσ = - 2 ∇⋅((∂ε/∂σ) ∇dρ)
    #
    # Note that below the LDA term (∂^2ε/∂ρ^2) dρ is ignored (already dealt with)
    ρ   = density.ρ
    ∇ρ  = density.∇ρ_real
    dρ  = perturbation.ρ
    ∇dρ = perturbation.∇ρ_real
    dσ = 2sum(∇ρ[α] .* ∇dρ[α] for α in 1:3)

    (
        terms.v2rhosigma .* dσ + divergence_real(density.basis) do α
            @. begin
                -2 * terms.v2rhosigma * ∇ρ[α] * dρ
                -2 * terms.v2sigma2 * ∇ρ[α] * dσ
                -2 * terms.vsigma * ∇dρ[α]
            end
        end
    )
end
