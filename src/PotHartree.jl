struct PotHartree
    """Plane-wave basis object"""
    basis::PlaneWaveBasis
end

struct PrecompHartree
    values_Yst  # Values on the Yst grid
end

"""
Construct an empty PrecompHartree object
"""
function PrecompHartree(basis::PlaneWaveBasis)
    PrecompHartree(similar(basis.grid_Yst, eltype(basis)))
end
empty_precompute(pot::PotHartree) = PrecompHartree(pot.basis)


function apply_real!(tmp_Yst, pot::PotHartree, precomp::PrecompHartree, in_Yst)
    tmp_Yst .= precomp.values_Yst .* in_Yst
end

"""
In-place compute some heavy stuff, which is needed for the
Hartree potential term of a particular SCF cycle.

Requires the density expressed in the density basis Y.
"""
function precompute!(precomp::PrecompHartree, pot::PotHartree, ρ_Y)
    pw = pot.basis

    # The |G|^2 values are cached inside PlaneWaveBasis in the
    # same order as the elements of ρ_Y. Thus computing the second
    # derivative (i.e. solving the Poission equation ΔV = -4π ρ
    # boils down to:
    values_Y = 4π * ρ_Y ./ pw.Gsq

    # Zero the DC component for the constant charge background:
    values_Y[pw.idx_DC] = 0

    # TODO Given that we cannot in-place compute into the
    #      real-type array we store in PrecompHartree, maybe it makes
    #      sense to switch to storing complex objects instead.

    # Fourier-transform and store in values_Yst
    T = eltype(pw)
    values_Yst = similar(precomp.values_Yst, Complex{T})
    Y_to_Yst!(pw, values_Y, values_Yst)

    if maximum(imag(values_Yst)) > 100 * eps(T)
        raise(ArgumentError("Expected potential on the real-space grid Y* to be entirely" *
                            " real-valued, but the present potential gives rise to a " *
                            "maximal imaginary entry of $(maximum(imag(values_Yst)))."))
    end
    precomp.values_Yst[:] = real(values_Yst)
    precomp
end
