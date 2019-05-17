"""
DFTK --- The density-functional toolkit

DFTK is a package of julia functions for playing with
plane-wave density-functional theory algorthms.

# Documentation
## Terminology and Definitions
General terminology used throughout the documentation
of the plane-wave aspects of the code.

e_G      Orthonormal plane wave
kpoints  Set of kpoints used for a calculation in the BZ.
E_cut    Plane-wave energy cutoff
α_Y      Supersampling for the potential basis
         (called supersampling_Y in the code)
α_Z      Supersampling for the XC basis
         (called supersampling_Z in the code)

## Basis sets
    a) The k-block orbital basis X_k, consisting of all
       plane-wave basis functions below the desired energy cutoff
       E_cut for each k point:
           X_k = { e_G : 1/2 |G + k|^2 ≤ E_cut }
    b) The orbital basis X, consisting of the collection
       of all X_k, i.e.
           X = { e_{G+k} : k ∈ kpoints and e_G ∈ X_k }
    c) The potential or density basis Y, consisting of all
       plane waves on which a potential needs to be know
       in order to be consistent with X. In practice
           Y = { e_G : 1/2 |G|^2 ≤ α_Y E_cut }
       where α_Y = 4 is required for a numerically exact
       result, since
           Y = { e_{G+G'} : ∃k e_G, e_{G'} ∈ X_k }.
    d) The basis Z, which is used for computing the application
       of the exchange-correlation potential operator to the
       density ρ, expressed in the basis Y:
           Z = { e_G : 1/2 |G|^2 ≤ α_Z E_cut }
       Since the exchange-correlation potential might involve
       arbitrary powers of the density ρ, a numerically exact
       computation of the Integral
           <e_G | V_XC(ρ) e_{G'}>
       for e_G, e_{G'} ∈ X_k requires α_Z to be infinite.
       In practice, α_Z = α_Y is usually chosen.

## Real-space grids
    a) Y*: Potential integration grid: Grid used for convolutions
       of a potential with the discretised representation of a
       DFT orbital. It's construction from Y is as follows:
       One takes the sphere of reciprocal space grid points
       described by Y and embeds it inside a cubic box with
       length √(2 α_Z E_cut). Then Y* is the IFFT-dual real-space grid.
    b) Z*: XC integration grid: Grid used for convolutions of
       exchange-correlation functional terms with the density
       or derivatives of it. Constructed from Z in the same way
       as Y* from Y.
"""
module DFTK

using Printf
using Markdown
using LinearAlgebra

include("constants.jl")

export PlaneWaveBasis
export substitute_kpoints!
export Y_to_Yst!
export Xk_to_Yst!
export Yst_to_Y!
export Yst_to_Xk!
include("PlaneWaveBasis.jl")

export PspHgh
include("PspHgh.jl")

export compute_density
include("compute_density.jl")

export Kinetic
include("Kinetic.jl")

export PotLocal
include("PotLocal.jl")

export PotHartree
include("PotHartree.jl")

export Hamiltonian
include("Hamiltonian.jl")

export PreconditionerKinetic
include("Preconditioner.jl")

export lobpcg
include("lobpcg.jl")

export self_consistent_field
include("self_consistent_field.jl")

end # module DFTK
