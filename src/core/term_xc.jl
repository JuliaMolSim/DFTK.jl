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


struct TermXc
    functionals         # Functional or functionals to be used
    supersampling::Int  # Supersampling for the XC grid
end

function (term::TermXc)(basis::PlaneWaveModel, energy::Union{Ref,Nothing}, potential;
                        ρ=nothing, kwargs...)

    # TODO This function is pretty messy ... any ideas for cleanup welcome

    @assert ρ !== nothing
    T = real(eltype(ρ))
    model = basis.model
    @assert model.spin_polarisation == :none "Only spin_polarisation == :none implemented."

    function ifft(x)
        tmp = G_to_r(basis, x)
        @assert(maximum(abs.(imag(tmp))) < 100 * eps(eltype(x)),
                "Imaginary part too large $(maximum(imag(tmp)))")
        real(tmp)
    end

    # If required compute contracted density gradient σ and gradient ∇ρ_real
    σ_real = nothing
    ∇ρ_real = nothing
    if any(xc.family == Libxc.family_gga for xc in op.functional)
        ∇ρ_real = [ifft(im * [(model.recip_lattice * G)[α] for G in basis_ρ(pw)] .* ρ)
                   for α in 1:3]
        # TODO The above assumes CPU arrays
        σ_real = sum(∇ρ_real[α] .* ∇ρ_real[α] for α in 1:3)
    end

    ρ_real = ifft(ρ)  # Density in real space

    # Initialisation
    potential !== nothing && (potential .= 0)
    Epp = nothing  # Energy per unit particle
    if energy !== nothing
        Epp = similar(ρ_real)
        energy[] = 0
    end

    # Use multiple dispatch of the next functions to avoid writing nested if-then branches
    add_xc!(xc, ::Val{Libxc.family_lda}, Epp, ::Nothing) = evaluate_lda!(xc, ρ_real, E=Epp)
    add_xc!(xc, ::Val{Libxc.family_gga}, Epp, ::Nothing) = evaluate_gga!(xc, ρ_real, E=Epp)
    function add_xc!(xc, ::Val{Libxc.family_lda}, Epp, potential)
        Vρ = similar(ρ_real)
        evaluate_lda!(xc, ρ_real, Vρ=Vρ, E=Epp)
        potential .+= Vρ
    end
    function add_xc!(xc, ::Val{Libxc.family_gga}, Epp, potential)
        Vρ = similar(ρ_real)
        Vσ = similar(ρ_real)
        evaluate_gga!(xc, ρ_real, σ_real, Vρ=Vρ, Vσ=Vσ, E=Epp)

        # TODO Check the literature how this expression comes about in detail.
        #      Following Richard Martin, Electronic stucture, p. 158, the XC potential
        #      can be split as:
        #        Vxc = ∂(ρ ϵ_{XC})/∂ρ - ∇( ∂(ρ ϵ_{XC})/∂(∇ρ) )
        #      Now the second term can be rewritten by the chain rule
        #        -∇( ∂(ρ ϵ_{XC})/∂(∇ρ) ) = -∇( ∂(ρ ϵ_{XC})/∂(|∇ρ|^2) ∂(|∇ρ|^2)/∂(∇ρ) )
        #                                = -∇( ∂(ρ ϵ_{XC})/∂(|∇ρ|^2) 2∇ρ )
        #    libxc yields
        #        Vρ === ∂(ρ ϵ_{XC})/∂ρ
        #        Vσ === ∂(ρ ϵ_{XC})/∂(|∇ρ|^2)

        gradterm = sum(1:3) do α
            # Compute term inside -∇(  ) in Fourier space
            Vσ∇ρ = 2r_to_G(basis, Vσ .* ∇ρ_real[α] .+ 0im)

            # take derivative
            im * [(model.recip_lattice * G)[α] for G in basis_ρ(pw)] .* Vσ∇ρ
        end
        potential .+= (Vρ - ifft(gradterm))
    end
    function add_xc!(xc, type, Epp, potential)
        error("Functional family $(string(xc.family)) not implemented.")
    end

    # Loop over all functionals, evaluate them and compute the energy
    for xc in op.functional
        add_xc!(xc, Val(xc.functional), Epp, potential)

        if energy !== nothing
            # Factor (1/2) to avoid double counting of electrons (see energy expression)
            # Factor 2 because α and β operators are identical for spin-restricted case
            dVol = pw.unit_cell_volume / prod(size(pw.FFT))
            energy[] += 2 * sum(Epp .* ρ_real) * dVol / 2
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
    TermXc([make_functional(f) for f in functional], supersampling)
end
