"""
Kinetic energy: 1/2 sum_n f_n ∫ |∇ψn|^2.
             or 1/2 sum_n f_n ∫ blowup_function(-i∇Ψ)^2 for modified kinetic term  
"""
Base.@kwdef struct Kinetic{F}
    scaling_factor::Real = 1
    blowup::F = BlowupKineticEnergy() # blow-up function to smooth energy bands. Defaut is x↦x^2.
end

(kin::Kinetic)(basis) = TermKinetic(basis, kin.scaling_factor, kin.blowup)
function Base.show(io::IO, kin::Kinetic)
    fac = isone(kin.scaling_factor) ? "" : ", scaling_factor=$scaling_factor"
    print(io, "Kinetic($fac)")
end

"""
Define different energy cutoff strategies. BlowupKineticEnergy corresponds
to standard kinetic energies.
ADD REFs
"""
struct BlowupKineticEnergy end
struct BlowupCHV end

(blowup::BlowupKineticEnergy)(x) = sum(abs2, x)

"""
Blow-up function as proposed in [REF paper Cancès, Hassan, Vidal to be submitted]
Parameters have been computed to ensure C^2 regularity of the energies bands
away from crossings and Lipschitz continuity at crossings.
"""
function (blowup::BlowupCHV)(x)
    @assert (0 ≤ x < 1) "The blow-up function is defined on [0,1)"
    # Define the three parts of the blow-up function
    g1(x) = sum(abs2, x)
    # C^2 interpolation part. The coefficients have been obtained by solving a
    # linear system.
    C = [7449.468883604184, -50230.90030429237, 135207.0860245428,
         -181568.08750464107, 121624.06499834878, -32503.74733526006]
    g2(x) = C'*((x * ones(6)) .^(0:5))
    # Blow-up part
    Ca(a) = (3/2)*(a^2)*(1-a)^(2)
    a_opti = 0.1792973510040141
    g3(x) = Ca(a_opti)/( (1-x)^(2) )

    # Assemble all parts
    x1, x2 = 0.7, 0.8
    (0 ≤ x < x1)   && return g1(x)
    (x1 ≤ x < x2)  && return g2(x)
    (x2 ≤ x < 1)   && return g3(x)
    (x==1) && return 1e6 # Handle |G+k|^2 = E_cut case
end

struct TermKinetic <: Term
    scaling_factor::Real  # scaling factor, absorbed into kinetic_energies
    # kinetic energies 1/2(k+G)^2 (or Ecut*blowup_function(|k+G|/√(2Ecut)) for energy
    # cutoff smearing methods) for each k-point
    kinetic_energies::Vector{<:AbstractVector}
end
function TermKinetic(basis::PlaneWaveBasis{T}, scaling_factor, blowup) where {T}
    kinetic_energies = [[T(scaling_factor) * basis.Ecut * blowup(norm(Gk)/√(2Ecut))*
                         for Gk in Gplusk_vectors_cart(basis, kpt)]
                        for kpt in basis.kpoints]
    TermKinetic(T(scaling_factor), kinetic_energies)
end

@timing "ene_ops: kinetic" function ene_ops(term::TermKinetic, basis::PlaneWaveBasis{T},
                                            ψ, occ; kwargs...) where {T}
    ops = [FourierMultiplication(basis, kpoint, term.kinetic_energies[ik])
           for (ik, kpoint) in enumerate(basis.kpoints)]
    isnothing(ψ) && return (E=T(Inf), ops=ops)

    E = zero(T)
    for (ik, k) in enumerate(basis.kpoints)
        for iband = 1:size(ψ[ik], 2)
            ψnk = @views ψ[ik][:, iband]
            E += (basis.kweights[ik] * occ[ik][iband]
                  * real(dot(ψnk, Diagonal(term.kinetic_energies[ik]), ψnk)))
        end
    end
    E = mpi_sum(E, basis.comm_kpts)

    (E=E, ops=ops)
end
