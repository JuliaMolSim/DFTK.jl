include("xc_evaluate.jl")

# Functionality for building the XC potential term and constructing the builder itself.

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


# TODO Integrate this with compute_density.jl
struct DensityDervatives
    basis
    max_derivative::Int
    ρ_real    # density on real-space grid
    ∇ρ_real   # density gradient on real-space grid
    σ_real    # contracted density gradient on real-space grid
end

"""
DOCME compute density in real space and its derivatives starting from Fourier-space density ρ
"""
function DensityDervatives(basis, max_derivative::Integer, ρ)
    model = basis.model
    @assert model.spin_polarisation == :none "Only spin_polarisation == :none implemented."
    function ifft(x)
        tmp = G_to_r(basis, x)
        @assert(maximum(abs.(imag(tmp))) < 100 * eps(real(eltype(x))),
                "Imaginary part too large $(maximum(imag(tmp)))")
        real(tmp)
    end

    σ_real = nothing
    ∇ρ_real = nothing
    if max_derivative < 0 || max_derivative > 1
        error("max_derivative not in [0, 1]")
    elseif max_derivative > 0
        ∇ρ_real = [ifft(im * [(model.recip_lattice * G)[α] for G in basis_Cρ(basis)] .* ρ)
                   for α in 1:3]
        # TODO The above assumes CPU arrays
        σ_real = sum(∇ρ_real[α] .* ∇ρ_real[α] for α in 1:3)
    end

    ρ_real = ifft(ρ .+ 0im)  # Density in real space, +0im to enforce complex algebra
    DensityDervatives(basis, max_derivative, ρ_real, ∇ρ_real, σ_real)
end

# Small internal helper function
epp_to_kwargs_(::Nothing) = Dict()
epp_to_kwargs_(Epp) = Dict(:E => Epp)

# Evaluate a single XC functional, *adds* the value to the potential on the grid
# and *returns* the energy per unit particle on the grid
# These are internal functions
eval_xc_!(basis, xc, family, Epp::Nothing, potential::Nothing, density...) = nothing
function eval_xc_!(basis, xc, ::Val{Libxc.family_lda}, Epp, ::Nothing, density)
    evaluate_lda!(xc, density.ρ_real; epp_to_kwargs_(Epp)...)
end
function eval_xc_!(basis, xc, ::Val{Libxc.family_gga}, Epp, ::Nothing, density)
    evaluate_gga!(xc, density.ρ_real, density.σ_real; epp_to_kwargs_(Epp)...)
end
function eval_xc_!(basis, xc, ::Val{Libxc.family_lda}, Epp, potential, density)
    Vρ = similar(density.ρ_real)
    evaluate_lda!(xc, density.ρ_real; Vρ=Vρ, epp_to_kwargs_(Epp)...)
    potential .+= Vρ
end
function eval_xc_!(basis, xc, ::Val{Libxc.family_gga}, Epp, potential, density)
    # Computes XC potential: Vρ - 2 div(Vσ ∇ρ)

    model = basis.model
    Vρ = similar(density.ρ_real)
    Vσ = similar(density.ρ_real)
    evaluate_gga!(xc, density.ρ_real, density.σ_real; Vρ=Vρ, Vσ=Vσ, epp_to_kwargs_(Epp)...)

    # 2 div(Vσ ∇ρ)
    gradterm = sum(1:3) do α
        # Compute term inside -∇(  ) in Fourier space
        Vσ∇ρ = 2r_to_G(basis, Vσ .* density.∇ρ_real[α] .+ 0im)

        # take derivative
        im * [(model.recip_lattice * G)[α] for G in basis_Cρ(basis)] .* Vσ∇ρ
    end
    gradterm_real = real(G_to_r(basis, gradterm))
    potential .+= (Vρ - gradterm_real)
end
function eval_xc_!(basis, xc, family, Epp, potential, density)
    error("Functional family $(string(xc.family)) not implemented.")
end

# Term structure, representing the exchange-correlation term
struct TermXc
    functionals         # Functional or functionals to be used
    supersampling::Int  # Supersampling for the XC grid
end

"""
Todo docme
"""
function (term::TermXc)(basis::PlaneWaveModel, energy::Union{Ref,Nothing}, potential;
                        ρ=nothing, kwargs...)
    @assert ρ !== nothing
    T = real(eltype(ρ))
    model = basis.model

    # Take derivatives of the density if needed.
    max_ρ_derivs = 0
    if any(xc.family == Libxc.family_gga for xc in term.functionals)
        max_ρ_derivs = 1
    end
    density = DensityDervatives(basis, max_ρ_derivs, ρ)

    # Initialisation
    potential !== nothing && (potential .= 0)
    Epp = nothing  # Energy per unit particle
    if energy !== nothing
        Epp = similar(density.ρ_real)
        energy[] = 0
    end

    for xc in term.functionals
        # *adds* the potential for this functional to `potential` and *sets* the
        # energy per unit particle in `Epp`.
        eval_xc_!(basis, xc, Val(xc.family), Epp, potential, density)

        if Epp !== nothing
            # Factor (1/2) to avoid double counting of electrons
            # Factor 2 because α and β operators are identical for spin-restricted case
            dVol = basis.model.unit_cell_volume / prod(basis.fft_size)
            energy[] += 2 * sum(Epp .* density.ρ_real) * dVol / 2
        end
    end
    energy, potential
end

"""
    term_xc(functionals; supersampling=2)

Construct an exchange-correlation term. `functionals` is a Libxc.jl `Functional`
objects to be used or its symbol or a list of such objects.
`supersampling` specifies the supersampling factor for the exchange-correlation
integration grid.
"""
function term_xc(functionals...; supersampling=2)
    @assert(supersampling == 2, "Only the case supersampling == 2 is implemented")
    # TODO Actually not even that ... we assume the grid to use is the density grid

    make_functional(func::Functional) = func
    make_functional(symb::Symbol) = Functional(symb)
    TermXc([make_functional(f) for f in functionals], supersampling)
end
