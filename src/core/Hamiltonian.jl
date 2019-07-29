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
function apply_hamiltonian!(out::AbstractVecOrMat, ham, ik::Int,
                            pot_hartree_values, pot_xc_values, in::AbstractVecOrMat)
    pw = ham.basis

    # Apply kinetic and non-local potential if given, accumulate results
    tmp = similar(out)
    apply_fourier!(out, ham.kinetic, ik, in)
    out .+= apply_fourier!(tmp, ham.pot_nonlocal, ik, in)

    fft_terms = [ham.pot_local, ham.pot_hartree, ham.pot_xc]
    if any(term !== nothing for term in fft_terms)
        # If any of the terms requiring an iFFT is present, do an iFFT
        in_real = similar(in, size(pw.FFT)..., size(in, 2))
        in_real = G_to_r!(pw, in, in_real, gcoords=pw.basis_wf[ik])

        # Apply the terms and accumulate
        accu_real = zero(in_real)
        tmp_real = similar(in_real)
        accu_real .+= apply_real!(tmp_real, ham.pot_local, in_real)
        accu_real .+= apply_real!(tmp_real, pot_hartree_values, in_real)
        accu_real .+= apply_real!(tmp_real, pot_xc_values, in_real)

        # FFT back to B_{Ψ,k} basis, accumulate, notice that this call
        # invalidates the data of accu_real as well.
        out .+= r_to_G!(pw, accu_real, tmp, gcoords=pw.basis_wf[ik])
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
Update the potential values of a non-linear term (e.g. `pot_hartree` or `pot_xc`)
on the real-space density grid ``B^∗_ρ``, given a current density `ρ` in the density basis
``B_ρ``. The updated values are returned as well.
"""
function update_potential!(potential, op, ρ)
    update_energies_potential!(Dict(), potential, op, ρ).potential
end

@doc raw"""
    update_energies_potential!(energies, potential, op, ρ)

Update the energies and the potential of a non-linear term (e.g. `pot_hartree` or `pot_xc`)
on the real-space density grid ``B^∗_ρ``, given a current density `ρ` in the density basis
``B_ρ``. If the passed term is `nothing`, `energies` and `potential` will be returned
as passed, else the appropriate keys in the `energies` dictionary and the values in the
potential array will be updated and the two objects returned.
"""
function update_energies_potential!(energies, potential, op::Nothing, ρ)
    (energies=energies, potential=potential)
end

@doc raw"""
Return an appropriately sized container for a potential term
on the real-space grid ``B^∗_ρ``
"""
empty_potential(op::Nothing) = nothing
empty_potential(op) = Array{eltype(op.basis.lattice)}(undef, size(op.basis.FFT)...)

"""
    update_energies_1e!(energies, ham, ρ, Psi, occupation)

Update the one-electron (linear) energy contributions of the Hamiltonian `ham`
inside the dictionary `energies`.
"""
function update_energies_1e!(energies, ham, ρ, Psi, occupation)
    update_energies_fourier!(energies, ham.kinetic,      Psi, occupation)
    update_energies_fourier!(energies, ham.pot_nonlocal, Psi, occupation)

    if ham.pot_local !== nothing
        pw = ham.basis
        ρ_real = similar(ρ, complex(eltype(ρ)), size(pw.FFT)...)
        G_to_r!(pw, ρ, ρ_real)
        ρ_real = real(ρ_real)
        update_energies_real!(energies, ham.pot_local, ρ_real)
    end

    energies
end

"""
    update_energies_fourier!(energies, op, Psi, occupation)

Update the energy contribution in the dictionary `energies` related to operator `op`
using the wave function `Psi` and the band numbers in `occupation`.
"""
update_energies_fourier!(energies, op::Nothing, Psi, occupation) = energies
function update_energies_fourier!(energies, op, Psi, occupation)
    pw = op.basis
    symbol = nameof(typeof(op))
    energies[symbol] = real(sum(
          wk * tr(Diagonal(occupation[ik])
                  * Psi[ik]' * apply_fourier!(similar(Psi[ik]), op, ik, Psi[ik]))
          for (ik, wk) in enumerate(pw.kweights)
    ))
    energies
end

"""
    update_energies_real!(energies, op, ρ_real)

Update the energy contribution in the dictionary `energies` related to operator `op`
using the real space density `ρ_real`.
"""
update_energies_real!(energies, op::Nothing, ρ_real) = energies
