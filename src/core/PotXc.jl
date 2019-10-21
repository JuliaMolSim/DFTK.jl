include("xc_evaluate.jl")

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
=#

struct PotXc
    basis::PlaneWaveBasis
    supersampling::Int  # Supersampling for the XC grid
    functional          # Functional or functionals to be used
end


"""
    PotXc(basis, functional, supersampling=2)

Construct an exchange-correlation term. `functional` is the Libxc.jl `Functional` to be used
or its symbol or a list of such objects. `supersampling` specifies the supersampling factor
for the exchang-correlation integration grid.
"""
function PotXc(basis::PlaneWaveBasis, functional...; supersampling=2)
    @assert(supersampling == 2, "Only the case supersampling == 2 is implemented")
    # TODO Actually not even that ... we assume the grid to use is the density grid
    make_functional(func::Functional) = func
    make_functional(symb::Symbol) = Functional(symb)
    PotXc(basis, supersampling, [make_functional(f) for f in functional])
end

function update_energies_potential!(energies, potential, op::PotXc, ρ)
    T = real(eltype(ρ))
    pw = op.basis

    # FFT density to real space
    ρ_real = similar(potential, Complex{T})
    G_to_r!(pw, ρ, ρ_real)
    @assert(maximum(abs.(imag(ρ_real))) < 100 * eps(T),
            "Imaginary part too large $(maximum(imag(ρ_real)))")
    ρ_real = real(ρ_real)

    # if required compute contracted density gradient σ
    σ_real = nothing
    ∇ρ_real = nothing
    if any(xc.family == Libxc.family_gga for xc in op.functional)
        ∇ρ = [vec([im * (pw.recip_lattice * G)[α] * ρ[ig]
                   for (ig, G) in enumerate(basis_ρ(pw))]) for α in 1:3]
        ∇ρ_real = [real(G_to_r!(pw, ∇ρ[α], similar(potential, Complex{T}))) for α in 1:3]
        σ_real = sum(∇ρ_real[α] .* ∇ρ_real[α] for α in 1:3)
    end

    potential .= 0
    E = similar(ρ_real)
    for xc in op.functional
        # Compute xc potential: Vρ - 2 div(Vσ ∇ρ)
        if xc.family == Libxc.family_lda
            Vρ = similar(ρ_real)
            evaluate_lda!(xc, ρ_real, Vρ=Vρ, E=E)
            potential .+= Vρ
        elseif xc.family == Libxc.family_gga
            Vρ = similar(ρ_real)
            Vσ = similar(ρ_real)
            evaluate_gga!(xc, ρ_real, σ_real, Vρ=Vρ, Vσ=Vσ, E=E)

            # 2 div(Vσ ∇ρ)
            gradterm = sum(1:3) do α
                # Compute term inside -∇(  ) in Fourier space and take derivative
                Vσ∇ρ = 2r_to_G!(pw, Vσ .* ∇ρ_real[α] .+ 0im, similar(ρ, Complex{T}))
                vec([im * (pw.recip_lattice * G)[α] * Vσ∇ρ[ig]
                     for (ig, G) in enumerate(basis_ρ(pw))])
            end

            gradterm_real = G_to_r!(pw, gradterm, similar(potential, Complex{T}))
            @assert(maximum(abs.(imag(gradterm_real))) < 100 * eps(T),
                    "Imaginary part too large $(maximum(imag(gradterm_real)))")
            potential .+= (Vρ - real(gradterm_real))
        else
            error("Functional family $(string(xc.family)) not implemented.")
        end

        # Factor (1/2) to avoid double counting of electrons (see energy expression)
        # Factor 2 because α and β operators are identical for spin-restricted case
        dVol = pw.unit_cell_volume / prod(size(pw.FFT))
        energies[xc.identifier] = 2 * sum(E .* ρ_real) * dVol / 2
    end

    (energies=energies, potential=potential)
end
