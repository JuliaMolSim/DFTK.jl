"""
Kinetic energy: 1/2 sum_n f_n ∫ Blowup_function(|∇ψn|).
"""
struct Kinetic
    scaling_factor::Real
    Blowup_function # blow-up fonction to smooth energy bands. Defaut is x↦x^2.
end
Kinetic(; scaling_factor=1, Blowup_function=x->sum(abs2, x)) = Kinetic(scaling_factor,
                                                                       Blowup_function)
(kin::Kinetic)(basis) = TermKinetic(basis, kin.scaling_factor, kin.Blowup_function)
function Base.show(io::IO, kin::Kinetic)
    fac = isone(kin.scaling_factor) ? "" : ", scaling_factor=$scaling_factor"
    print(io, "Kinetic($fac)")
end

struct TermKinetic <: Term
    scaling_factor::Real  # scaling factor, absorbed into kinetic_energies
    # kinetic energy Ecut*Blowup_function(|G+k|/√(2Ecut)) for every kpoint
    kinetic_energies::Vector{<:AbstractVector}
end
function TermKinetic(basis::PlaneWaveBasis{T}, scaling_factor, Blowup_function) where {T}
    Ecut = basis.Ecut
    kinetic_energies = [[T(scaling_factor) * Ecut * Blowup_function(norm(Gk)/√(2*Ecut))
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

"""
Blow-up function as proposed in [REF paper Cancès, Hassan, Vidal to be submitted]
"""
function Blowup_function_CHV()
    g1(x) = sum(abs2, x)
    # C^2 interpolation part
    C = [7449.468883604184, -50230.90030429237, 135207.0860245428,
         -181568.08750464107, 121624.06499834878, -32503.74733526006]
    g2(x) = C'*((x * ones(6)) .^(0:5))
    # Blow-up part
    Ca(a) = (3/2)*(a^2)*(1-a)^(2)
    a_opti = 0.1792973510040141
    g3(x) = Ca(a_opti)/( (1-x)^(2) )

    # Assemble all parts
    function Blowup_function(x)
        x1, x2 = [0.7, 0.8]
        (0 ≤ x < x1)   && (return g1(x))
        (x1 ≤ x < x2)  && (return g2(x))
        (x2 ≤ x < 1)   && (return g3(x))
        (x==1) && (return 1e6) # Handle |G+k|^2 = E_cut case
        error("The blow-up function is defined on [0,1). Did you devide by √Ecut ?")
    end
    Blowup_function
end

"""
Blow-up function as implemented in the Abinit code [ref code].
"""
function Blowup_function_Abinit(Ecut, Ecutsm)
    function Blowup_function(x)
        kin_x = sum(abs2, x)
        (0 ≤ x ≤ Ecutsm/Ecut) && (return kin_x)
        (x > 1) && (error("The blow-up function is defined on [0,1)."*
                          " Did you devide by √Ecut ?"))
        y = Ecut*(1-kin_x)/Ecutsm
        return kin_x/(y^2*(3+y-6*y^2+3*y^3))
    end
    Blowup_function
end
