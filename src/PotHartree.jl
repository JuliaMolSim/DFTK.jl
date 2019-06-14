struct PotHartree
    basis::PlaneWaveBasis
end

"""
In-place compute some heavy stuff, which is needed for the
Hartree potential term of a particular SCF cycle.

Requires the density expressed in the density basis Y.
"""
function compute_potential!(precomp, pot::PotHartree, ρ_Y)
    pw = pot.basis

    # The |G|^2 values are cached inside PlaneWaveBasis in the
    # same order as the elements of ρ_Y. Thus computing the second
    # derivative (i.e. solving the Poission equation ΔV = -4π ρ
    # boils down to:
    Gsq = [sum(abs2, pw.recip_lattice * G) for G in gcoords(pw)]
    values_Y = 4π * ρ_Y ./ Gsq

    # Zero the DC component for the constant charge background:
    values_Y[pw.idx_DC] = 0

    # TODO Given that we cannot in-place compute into the
    #      real-type array we store in PrecompHartree, maybe it makes
    #      sense to switch to storing complex objects instead.

    # Fourier-transform and store in values_real
    T = eltype(pw.lattice)
    values_real = similar(precomp, Complex{T})
    G_to_R!(pw, values_Y, values_real)

    if maximum(imag(values_real)) > 100 * eps(T)
        raise(ArgumentError("Expected potential on the real-space grid Y* to be entirely" *
                            " real-valued, but the present potential gives rise to a " *
                            "maximal imaginary entry of $(maximum(imag(values_real)))."))
    end
    precomp .= real(values_real)
end
