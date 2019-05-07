"""
Structure identifying a Plane-Wave-discretised Hamiltonian.

The Hamiltonian is effectively a collection of terms and
can be applied in a k-Point block using `apply_Xk!`.
"""
struct Hamiltonian
    """The plane-wave basis X used to discretise the Hamiltonian"""
    basis

    """
    The kinetic term, typically a scaled Laplacian, -1/2 Δ
    It is expected that this object supports a
    ```
    apply_Xk!(out, kinetic, k, in)
    ```
    function call to compute its action at a particular k point.
    """
    kinetic

    """
    Term representing a local potential
    It is expected that this object supports a
    ```
    apply_Yst!(out, pot_local, in)
    ```
    function call, which should compute the action of the potential
    on the grid Y*. The potential should not depend on the density
    or the SCF orbitals.
    """
    pot_local

    """
    Term representing a non-local potential
    This object should support a
    ```
    apply_Xk!(out, pot_nonlocal, k, in)
    ```
    function call to compute its action at a particular k point.
    """
    pot_nonloc

    """
    Term representing the Hartree potential
    This term will be applied on the grid Y* using the signature
    ```
    apply_Yst!(out, pot_local, precomp, in)
    ```
    where precomp is precomputed data which was obtained earlier
    by calling `precompute` on the object
    (e.g. `precompute(pot_hartree, ρ_Y)`).
    """
    pot_hartree

    """
    Term representing the exchange-correlation potential
    This term will be applied on the grid Y*, similar to
    pot_hartree, i.e. with a signature
    ```
    apply_Yst!(out, pot_local, precomp, in)
    ```
    """
    pot_xc
end

"""
    apply_Xk!(out_k, ham, k, precomp_hartree, precomp_xc, X_k)

Apply a k-point block of a Hamiltonian using precomputed data for the
Hartree and XC terms.

The application proceeds as follows:
    - The kinetic and non-local potential terms are applied directly 
      in the plane-wave basis.
    - For the other terms we require the respective potential `V` on Y*
      and proceed using the protocol
      ```text
        X_k --pad--> Y --IFFT--> Y* --V--> Y* --FFT--> Y --trunc--> X_k
      ```
      where the meaning is
      ```text
        pad      Zero padding
        FFT      fast-Fourier transform
        IFFT     inverse fast-Fourier transform
        trunc    Truncation to a smaller basis
        V        Apply potential elementwise.
      ```
"""
function apply_Xk!(out_k, ham::Hamiltonian, k, precomp_hartree, precomp_xc, X_k)
    # TODO Pseudocode:
    #=
    Yst = pad_and_ifft(X_k)

    resloc = Array{ComplexF64}(appropriate_size)
    reshart = Array{ComplexF64}(appropriate_size)
    resxc = Array{ComplexF64}(appropriate_size)
    apply_Yst!(resloc, pot_loc, Yst)
    apply_Yst!(reshart, pot_hartree, pot_hartree_precomp, Yst)
    apply_Yst!(resxc, pot_xc, pot_xc_precomp, Yst)
    resYst = resloc .+ reshart .+ resxc
    resXk = fft_and_trunc(resYst)

    resnloc = Array{ComplexF64}(appropriate_size)
    reskin = Array{ComplexF64}(appropriate_size)
    apply_Xk!(resnloc, pot_nonloc, k, X_k)
    apply_Xk!(reskin, kinetic, k, X_k)

    out_k .= resXk .+ resnloc .+ reskin
    =#
end
