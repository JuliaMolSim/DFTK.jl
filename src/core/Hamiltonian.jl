struct Hamiltonian
    basis         # Plane-wave basis object
    kinetic       # Kinetic energy operator
    pot_local     # Local potential term
    pot_nonlocal  # Non-local potential term
    pot_hartree   # Hartree potential term
    pot_xc        # Exchange-correlation term
end


"""
    Hamiltonian(basis::PlaneWaveBasis; kinetic=Kinetic(basis), pot_local=nothing,
                pot_nonlocal=nothing, pot_hartree=nothing, pot_xc=nothing)

Hamiltonian discretized in a plane-wave `basis`. Terms which are not specified
(left as `nothing`) are ignored during application. If only a basis object is specified,
a free-electron Hamiltonian is constructed. The `kinetic` `pot_local` and `pot_nonlocal`
terms shall not contain a non-linearity in the density.
"""
function Hamiltonian(basis; kinetic=Kinetic(basis), pot_local=nothing, pot_nonlocal=nothing,
            pot_hartree=nothing, pot_xc=nothing)
    Hamiltonian(basis, kinetic, pot_local, pot_nonlocal, pot_hartree, pot_xc)
end


@doc raw"""
    apply_hamiltonian!(out, ham, ik, pot_hartree_values, pot_xc_values, in)

Apply the `ik`-th ``k``-Point block of a Hamiltonian using precomputed values
for the non-linear Hartree and XC potential (in `pot_hartree_values` and
`pot_xc_values`) on the real-space grid ``B_\rho^∗`` at the current density.

For this `kinetic` and `pot_nonlocal` are applied to `in` using `apply_fourier!`.
The other terms are treated via a convolution on the density grid ``B_\rho^∗``,
that is the procedure
```
 X_k --pad--> Y --IFFT--> Y* --V--> Y* --FFT--> Y --trunc--> X_k
```
where
```
  pad      Zero padding
  FFT      fast-Fourier transform
  IFFT     inverse fast-Fourier transform
  trunc    Truncation to a smaller basis
  V        Apply potential elementwise using `apply_real!`
```
"""
function apply_hamiltonian!(out::AbstractVector, ham, ik::Int,
                            pot_hartree_values, pot_xc_values, in::AbstractVector)
    pw = ham.basis

    # Apply kinetic and non-local potential if given, accumulate results
    tmp = similar(out)
    apply_fourier!(out, ham.kinetic, ik, in)
    out .+= apply_fourier!(tmp, ham.pot_nonlocal, ik, in)

    fft_terms = [ham.pot_local, ham.pot_hartree, ham.pot_xc]
    if any(term !== nothing for term in fft_terms)
        # If any of the terms requiring an iFFT is present, do an iFFT
        in_real = similar(in, size(pw.FFT)...)
        in_real = G_to_R!(pw, in, in_real, gcoords=pw.wf_basis[ik])

        # Apply the terms and accumulate
        accu_real = zero(in_real)
        tmp_real = similar(in_real)
        accu_real .+= apply_real!(tmp_real, ham.pot_local.values_real, in_real)
        accu_real .+= apply_real!(tmp_real, pot_hartree_values, in_real)
        accu_real .+= apply_real!(tmp_real, pot_xc_values, in_real)

        # FFT back to B_{Ψ,k} basis, accumulate, notice that this call
        # invalidates the data of accu_real as well.
        out .+= R_to_G!(pw, accu_real, tmp, gcoords=pw.wf_basis[ik])
    end
    out
end
function apply_hamiltonian!(out, ham, ik::Int, pot_hartree_values, pot_xc_values, in)
    # TODO This a fix for now to get it to work
    #      Ideally the above function should be able to deal with this directly
    n_bas, n_vec = size(in)
    for iv in 1:n_vec
        apply_hamiltonian!(view(out, :, iv), ham, ik, pot_hartree_values, pot_xc_values,
                           view(in, :, iv))
    end
    out
end

Base.eltype(ham::Hamiltonian) = Complex{eltype(ham.basis.lattice)}


@doc raw"""Apply a ``k``-block of a Hamiltonian term in Fourier space"""
apply_fourier!(out, op::Nothing, ik::Int, in) = (out .= 0)


@doc raw"""
Apply a Hamiltonian term by computation on the real-space density grid ``B^∗_ρ``
"""
apply_real!(out, values::Nothing, in) = (out .= 0)
apply_real!(out, values, in) = (out .= values .* in)

@doc raw"""
Compute the potential of a non-linear term (e.g. `pot_hartree` or `pot_xc`)
on the real-space density grid ``B^∗_ρ``, given a current density `ρ`
in the density basis ``B_ρ``. If the passed term is `nothing`,
`nothing` is returned by the function as well, else an array of values.
"""
compute_potential!(precomp, op::Nothing, ρ) = nothing

@doc raw"""
Return an appropriately sized container for a potential term
on the real-space grid ``B^∗_ρ``
"""
empty_potential(op::Nothing) = nothing
empty_potential(op) = Array{eltype(op.basis.lattice)}(undef, size(op.basis.FFT)...)
