# Data structures

```@setup data_structures
using DFTK
a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

model = model_LDA(lattice, atoms)
kgrid = [4, 4, 4]
Ecut = 15
basis = PlaneWaveBasis(model; Ecut, kgrid)
scfres = self_consistent_field(basis, tol=1e-8);
```

In this section we assume a calculation of silicon LDA model
in the setup described in [Tutorial](@ref).

## `Model` datastructure
The physical model to be solved is defined by the `Model`
datastructure. It contains the unit cell, number of electrons, atoms,
type of spin polarization and temperature. Each atom has an atomic
type (`Element`) specifying the number of valence electrons and the
potential (or pseudopotential) it creates with respect to the electrons.
The `Model` structure also contains the list of energy terms
defining the energy functional to be minimised during the SCF.
For the silicon example above, we used
an LDA model, which consists of the following terms[^2]:

[^2]: If you are not familiar with Julia syntax, `typeof.(model.term_types)` is equivalent to `[typeof(t) for t in model.term_types]`.

```@example data_structures
typeof.(model.term_types)
```

DFTK computes energies for all terms of the model individually,
which are available in `scfres.energies`:

```@example data_structures
scfres.energies
```

For now the following energy terms are available in DFTK:

- Kinetic energy
- Local potential energy, either given by analytic potentials or
  specified by the type of atoms.
- Nonlocal potential energy, for norm-conserving pseudopotentials
- Nuclei energies (Ewald or pseudopotential correction)
- Hartree energy
- Exchange-correlation energy
- Power nonlinearities (useful for Gross-Pitaevskii type models)
- Magnetic field energy
- Entropy term

Custom types can be added if needed. For examples see
the definition of the above terms in the
[`src/terms`](https://dftk.org/tree/master/src/terms) directory.

By mixing and matching these terms, the user can create custom models
not limited to DFT. Convenience constructors are provided for common cases:

- `model_LDA`: LDA model using the
  [Teter parametrisation](https://doi.org/10.1103/PhysRevB.54.1703)
- `model_DFT`: Assemble a DFT model using
   any of the LDA or GGA functionals of the
   [libxc](https://tddft.org/programs/libxc/functionals/) library,
   for example:
   ```
   model_DFT(lattice, atoms, [:gga_x_pbe, :gga_c_pbe])
   model_DFT(lattice, atoms, :lda_xc_teter93)
   ```
   where the latter is equivalent to `model_LDA`.
   Specifying no functional is the reduced Hartree-Fock model:
   ```
   model_DFT(lattice, atoms, [])
   ```
- `model_atomic`: A linear model, which contains no electron-electron interaction
  (neither Hartree nor XC term).

## `PlaneWaveBasis` and plane-wave discretisations

The `PlaneWaveBasis` datastructure handles the discretization of a
given `Model` in a plane-wave basis.
In plane-wave methods the discretization is twofold:
Once the ``k``-point grid, which determines the sampling
*inside* the Brillouin zone and on top of that a finite
plane-wave grid to discretise the lattice-periodic functions.
The former aspect is controlled by the `kgrid` argument
of `PlaneWaveBasis`, the latter is controlled by the
cutoff energy parameter `Ecut`:

```@example data_structures
PlaneWaveBasis(model; Ecut, kgrid)
```

The `PlaneWaveBasis` by default uses symmetry to reduce the number of
`k`-points explicitly treated. For details see
[Crystal symmetries](@ref).

As mentioned, the periodic parts of Bloch waves are expanded
in a set of normalized plane waves ``e_G``:
```math
\begin{aligned}
  \psi_{k}(x) &= e^{i k \cdot x} u_{k}(x)\\
  &= \sum_{G \in \mathcal R^{*}} c_{G}  e^{i  k \cdot  x} e_{G}(x)
\end{aligned}
```
where ``\mathcal R^*`` is the set of reciprocal lattice vectors.
The ``c_{{G}}`` are ``\ell^{2}``-normalized. The summation is truncated to a
"spherical", ``k``-dependent basis set
```math
  S_{k} = \left\{G \in \mathcal R^{*} \,\middle|\,
          \frac 1 2 |k+ G|^{2} \le E_\text{cut}\right\}
```
where ``E_\text{cut}`` is the cutoff energy.

Densities involve terms like ``|\psi_{k}|^{2} = |u_{k}|^{2}`` and
therefore products ``e_{-{G}} e_{{G}'}`` for ``{G}, {G}'`` in
``S_{k}``. To represent these we use a "cubic", ``k``-independent
basis set large enough to contain the set
``\{{G}-G' \,|\, G, G' \in S_{k}\}``.
We can obtain the coefficients of densities on the
``e_{G}`` basis by a convolution, which can be performed efficiently
with FFTs (see [`G_to_r`](@ref) and [`r_to_G`](@ref) functions).
Potentials are discretized on this same set.

The normalization conventions used in the code is that quantities
stored in reciprocal space are coefficients in the ``e_{G}`` basis,
and quantities stored in real space use real physical values.
This means for instance that wavefunctions in the real space grid are
normalized as ``\frac{|\Omega|}{N} \sum_{r} |\psi(r)|^{2} = 1`` where
``N`` is the number of grid points.

For example let us check the normalization of the first eigenfunction
at the first ``k``-point in reciprocal space:

```@example data_structures
ψtest = scfres.ψ[1][:, 1]
sum(abs2.(ψtest))
```

We now perform an IFFT to get ψ in real space. The ``k``-point has to be
passed because ψ is expressed on the ``k``-dependent basis.
Again the function is normalised:

```@example data_structures
ψreal = G_to_r(basis, basis.kpoints[1], ψtest)
sum(abs2.(ψreal)) * basis.dvol
```

The list of ``k`` points of the basis can be obtained with `basis.kpoints`.

```@example data_structures
basis.kpoints
```

The ``G`` vectors of the "spherical", ``k``-dependent grid can be obtained
with `G_vectors`:

```@example data_structures
[length(G_vectors(kpoint)) for kpoint in basis.kpoints]
```

```@example data_structures
ik = 1
G_vectors(basis.kpoints[ik])[1:4]
```

The list of ``G`` vectors (Fourier modes) of the "cubic", ``k``-independent basis
set can be obtained similarly with `G_vectors(basis)`.

```@example data_structures
length(G_vectors(basis)), prod(basis.fft_size)
```

```@example data_structures
collect(G_vectors(basis))[1:4]
```

Analogously the list of ``r`` vectors
(real-space grid) can be obtained with `r_vectors(basis)`:

```@example data_structures
length(r_vectors(basis))
```

```@example data_structures
collect(r_vectors(basis))[1:4]
```

## Accessing Bloch waves and densities
Wavefunctions are stored in an array `scfres.ψ` as `ψ[ik][iG, iband]` where
`ik` is the index of the ``k``-point (in `basis.kpoints`), `iG` is the
index of the plane wave (in `G_vectors(basis.kpoints[ik])`) and
`iband` is the index of the band.
Densities are stored in real space, as a 4-dimensional array (the third being the spin component).

```@example data_structures
using Plots  # hide
rvecs = collect(r_vectors(basis))[:, 1, 1]  # slice along the x axis
x = [r[1] for r in rvecs]                   # only keep the x coordinate
plot(x, scfres.ρ[:, 1, 1, 1], label="", xlabel="x", ylabel="ρ", marker=2)
```

```@example data_structures
G_energies = [sum(abs2.(model.recip_lattice * G)) ./ 2 for G in G_vectors(basis)][:]
scatter(G_energies, abs.(r_to_G(basis, scfres.ρ)[:]);
        yscale=:log10, ylims=(1e-12, 1), label="", xlabel="Energy", ylabel="|ρ|^2")
```

Note that the density has no components on wavevectors above a certain energy,
because the wavefunctions are limited to ``\frac 1 2|k+G|^2 ≤ E_{\rm cut}``.
