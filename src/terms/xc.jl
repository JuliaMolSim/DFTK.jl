using Libxc
include("../xc/xc_evaluate.jl")

"""
Exchange-correlation term, defined by a list of functionals and usually evaluated through libxc.
"""
struct Xc
    functionals::Vector{Symbol}  # Symbols of the functionals (Libxc.jl / libxc convention)
    scaling_factor::Real         # to scale by an arbitrary factor (useful for exploration)
end
Xc(symbols::Vector; scaling_factor=1) = Xc(convert.(Symbol, symbols), scaling_factor)
Xc(symbols::Symbol...; kwargs...) = Xc([symbols...]; kwargs...)

function (xc::Xc)(basis)
    functionals = Functional.(xc.functionals; n_spin=basis.model.n_spin_components)
    TermXc(basis, functionals, xc.scaling_factor)
end

struct TermXc <: Term
    basis::PlaneWaveBasis
    functionals::Vector{Functional}
    scaling_factor::Real
end

function ene_ops(term::TermXc, ψ, occ; ρ, ρspin=nothing, kwargs...)
    basis = term.basis
    T     = eltype(basis)
    model = basis.model
    n_spin = model.n_spin_components
    @assert all(xc.family in (:lda, :gga) for xc in term.functionals)

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
        for σ in 1:n_spin
            potential[σ] .+= @view terms.vrho[σ, :, :, :]
        end

        if haskey(terms, :vsigma)
            # TODO Potential to save some FFTs for spin-polarised calculations:
            #      For n_spin == 2 this calls divergence_real 4 times, where only 2 are
            #      needed (one for potential[1] and one for potential[2])
            @views for σ in 1:n_spin, τ in 1:n_spin
                iστ = density.index_spin_σ[(σ, τ)]
                # Extra factor (1/2) for σ != τ is needed because libxc only keeps σ_{αβ}
                # in the energy expression. See comment block below on spin-polarised XC.
                spinfac = (σ == τ ? 2 : 1)
                potential[σ] .+=
                     -spinfac .* divergence_real(
                        α -> terms.vsigma[iστ, :, :, :] .* density.∇ρ_real[τ, :, :, :, α],
                        density.basis,
                    )
            end
        end
    end
    if term.scaling_factor != 1
        E *= term.scaling_factor
        potential = [pot .*= term.scaling_factor for pot in potential]
    end

    ops = [RealSpaceMultiplication(basis, kpoint, potential[kpoint.spin])
           for kpoint in basis.kpoints]
    (E=E, ops=ops)
end


function compute_kernel(term::TermXc; ρ::RealFourierArray, kwargs...)
    @assert term.basis.model.spin_polarization in (:none, :spinless)
    ρ_real = reshape(ρ.real, 1, size(ρ.real)...)
    kernel = similar(ρ.real)
    kernel .= 0
    for xc in term.functionals
        xc.family == :lda || error("compute_kernel only implemented for LDA")
        terms = evaluate(xc; rho=ρ_real, derivatives=2:2)  # only valid for LDA
        kernel .+= @view terms.v2rho2[1, :, :, :]
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
        result .+= @view(terms.v2rho2[1, :, :, :]) .* dρ.real
        if haskey(terms, :v2rhosigma)
            result .+= apply_kernel_term_gga(terms, density, perturbation)
        end
    end
    from_real(basis, term.scaling_factor .* result)
end

#=
The total energy is
Etot = ∫ ρ E(ρ,σ), where σ = |∇ρ|^2 and E(ρ,σ) is the energy per unit particle.
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
(boundary terms drop by periodicity). Therefore,
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

    dρ  = 2 IF(Xϕ) .* IF(X dϕ)
    d∇ρ = IF K F dρ
        = 2 IF K F (IF(Xϕ) .* IF(X dϕ))

    Etot  = sum(ρ . * E(ρ, ∇ρ.^2))
    dEtot = sum(Vρ .* dρ + 2 (Vσ .* ∇ρ) .* d∇ρ)
          = sum(Vρ .* dρ + 2 (Vσ .* ∇ρ) .* [(IF K F) dρ)]
          = sum(Vρ .* dρ - 2 [(IF K F) (Vσ .* ∇ρ)] .* dρ)
where we used that (IF K F) is anti-self-adjoint and thought of sum(A .* op B )
like ⟨A | op B⟩; this is the discrete integration by parts. Now note that

    sum(V .* dρ) = 2 sum(V .* IF(Xϕ) .* IF(X dϕ))
                 = 2 sum(R(F[ V .* IF(Xϕ) ]) .* dϕ)

where we took the adjoint (IF X)^† = (R F) and the result follows.
=#

#=  Spin-polarised calculations

These expressions can be generalised for spin-polarised calculations.
In this case for example the energy per unit particle becomes
E(ρ_α, ρ_β, σ_αα, σ_αβ, σ_βα, σ_ββ), where σ_ij = ∇ρ_i ⋅ ∇ρ_j
and the XC potential is analogously
    Vxc_s = Vρ_s - 2 ∑_t div(Vσ_{st} ∇ρ_t)
where s, t ∈ {α, β} are the spin components and we understand
    Vρ_s     = ∂(ρ E)/∂(ρ_s)
    Vσ_{s,t} = ∂(ρ E)/∂(σ_{s,t})

Now, in contrast to this libxc explicitly uses the symmetry σ_αβ = σ_βα and sets σ
to be a vector of the three independent components only
    σ = [σ_αα, σ_x, σ_ββ]  where     σ_x = (σ_αβ + σ_βα)/2
Accordingly Vσ has the compoments
    [∂(ρ E)/∂σ_αα, ∂(ρ E)/∂σ_x, ∂(ρ E)/∂σ_ββ]
where in particular ∂(ρ E)/∂σ_x = (1/2) ∂(ρ E)/∂σ_αβ = (1/2) ∂(ρ E)/∂σ_βα.
This explains the extra factor (1/2) needed in the GGA term of the XC potential.
=#

function max_required_derivative(functional)
    functional.family == :lda && return 0
    functional.family == :gga && return 1
    error("Functional family $(functional.family) not known.")
end

struct DensityDerivatives  # TODO Rename to LibxcDensity as libxc-specific, refactor
    basis
    max_derivative::Int
    ρ         # density on real-space grid
    ∇ρ_real   # density gradient on real-space grid
    σ_real    # contracted density gradient on real-space grid
    index_spin_σ  # Dict mapping a pair of spins to the index on the spin axis in σ
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

    n_spin    = model.n_spin_components
    σ_real    = nothing
    ∇ρ_real   = nothing
    if model.spin_polarization == :collinear
        ρ_real    = similar(ρ.real,    n_spin, basis.fft_size...)
        ρ_real[1, :, :, :] .= @. (ρ.real + ρspin.real) / 2
        ρ_real[2, :, :, :] .= @. (ρ.real - ρspin.real) / 2

        if max_derivative > 0
            ρ_fourier = similar(ρ.fourier, n_spin, basis.fft_size...)
            ρ_fourier[1, :, :, :] .= @. (ρ.fourier + ρspin.fourier) / 2
            ρ_fourier[2, :, :, :] .= @. (ρ.fourier - ρspin.fourier) / 2
        end

        index_spin_σ = Dict((1, 1) => 1, (1, 2) => 2, (2, 1) => 2, (2, 2) => 3)
        @assert n_spin == 2
    else
        ρ_real    = reshape(ρ.real,    1, basis.fft_size...)
        ρ_fourier = reshape(ρ.fourier, 1, basis.fft_size...)
        index_spin_σ = Dict((1, 1) => 1)
        @assert n_spin == 1
    end

    if max_derivative > 0
        n_spin_σ = div((n_spin + 1) * n_spin, 2)
        ∇ρ_real = similar(ρ.real,   n_spin, basis.fft_size..., 3)
        σ_real  = similar(ρ.real, n_spin_σ, basis.fft_size...)

        for α = 1:3
            iGα = [im * G[α] for G in G_vectors_cart(basis)]
            for σ = 1:n_spin
                ∇ρ_real[σ, :, :, :, α] .= ifft(iGα .* @view ρ_fourier[σ, :, :, :])
            end
        end

        for σ = 1:n_spin, τ=σ:n_spin
            iστ = index_spin_σ[(σ, τ)]
            σ_real[iστ, :, :, :] .= 0
            @views for α in 1:3
                σ_real[iστ, :, :, :] .+= ∇ρ_real[σ, :, :, :, α] .* ∇ρ_real[τ, :, :, :, α]
            end
        end
    end

    DensityDerivatives(basis, max_derivative, ρ_real, ∇ρ_real, σ_real, index_spin_σ)
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
    # dV(r) = (∂^2ε/∂ρ^2) dρ - 2 ∇⋅[(∂^2ε/∂σ∂ρ) ∇ρ + (∂ε/∂σ) (∂∇ρ/∂ρ)] dρ
    #       + (∂^2ε/∂ρ∂σ) dσ - 2 ∇⋅[(∂^ε/∂σ^2) ∇ρ  + (∂ε/∂σ) (∂∇ρ/∂σ)] dσ
    #
    # Note dσ = 2∇ρ d∇ρ = 2∇ρ ∇dρ, therefore
    #    - 2 ∇⋅((∂ε/∂σ) (∂∇ρ/∂σ)) dσ = - 2 ∇⋅((∂ε/∂σ) ∇dρ)
    # and (because assumed independent variables): (∂∇ρ/∂ρ) = 0.
    #
    # Note that below the LDA term (∂^2ε/∂ρ^2) dρ is ignored (already dealt with)
    ρ   = density.ρ
    ∇ρ  = density.∇ρ_real
    dρ  = perturbation.ρ
    ∇dρ = perturbation.∇ρ_real
    dσ = 2sum(∇ρ[α] .* ∇dρ[α] for α in 1:3)

    @views begin
        terms.v2rhosigma[1, :, :, :] .* dσ + divergence_real(density.basis) do α
            @. begin
                -2 * terms.v2rhosigma[1, :, :, :] * ∇ρ[α] * dρ
                -2 * terms.v2sigma2[1, :, :, :]   * ∇ρ[α] * dσ
                -2 * terms.vsigma[1, :, :, :]     * ∇dρ[α]
            end
        end
    end
end
