# Introduction to density-functional theory

!!! note "More details"
    This chapter only gives a very rough overview for now. Further details
    can be found in the [summary of DFT theory](https://michael-herbst.com/teaching/2022-mit-workshop-dftk/2022-mit-workshop-dftk/DFT_Theory.pdf)
    or in the [Introductory Resources](@ref introductory-resources).

Density-functional theory is a simplification of the full electronic
Schrödinger equation leading to an effective single-particle model.
Mathematically this can be cast as an energy minimisation problem
in the electronic density $\rho$. In the Kohn-Sham variant
of density-functional theory the corresponding first-order stationarity
conditions can be written as the non-linear eigenproblem
```math
\begin{aligned}
&\left( -\frac12 \Delta + V\left(\rho\right) \right) \psi_i = \varepsilon_i \psi_i, \\
V(\rho) = &\, V_\text{nuc} + V_\text{H}^\rho + V_\text{XC}(\rho), \\
\rho = &\sum_{i=1}^N f(\varepsilon_i)  \abs{\psi_i}^2, \\
\end{aligned}
```
where $\{\psi_1,\ldots, \psi_N\}$ are $N$ orthonormal orbitals,
the one-particle eigenfunctions and $f$ is the occupation function
(e.g. [`DFTK.Smearing.FermiDirac`](@ref), [`DFTK.Smearing.Gaussian`](@ref)
or [`DFTK.Smearing.MarzariVanderbilt`](@ref))
chosen such that the integral of the density is equal to the number of electrons in the system.
Further the potential terms that make up $V(\rho)$ are
- the nuclear attraction potential $V_\text{nuc}$ (interaction of electrons and nuclei)
- the exchange-correlation potential $V_\text{xc}$,
  depending on $\rho$ and potentially its derivatives.
- The Hartree potential $V_\text{H}^\rho$, which is obtained as the unique zero-mean solution to the periodic Poisson equation
  ```math
  -\Delta V_\text{H}^\rho(r)
  = 4\pi \left(\rho(r) - \frac{1}{|\Omega|} \int_\Omega \rho \right).
  ```
The non-linearity is such due to the fact that the DFT Hamiltonian
```math
H = -\frac12 \Delta + V\left(\rho\right)
```
depends on the density $\rho$, which is built from its own eigenfunctions.
Often $H$ is also called the Kohn-Sham matrix or Fock matrix.

Introducing the *potential-to-density map*
```math
D(V) = \sum_{i=1}^N f(\varepsilon_i)  \abs{\psi_i}^2
\qquad \text{where} \qquad \left(-\frac12 \Delta + V\right) \psi_i = \varepsilon_i \psi_i
```
allows to write the DFT problem in the short-hand form
```math
\rho = D(V(\rho)),
```
which is a fixed-point problem in $\rho$, also known as the
**self-consistent field problem** (SCF problem).
Notice that computing $D(V)$ requires the diagonalisation of the operator
$-\frac12 \Delta + V$, which is usually the most
expensive step when solving the SCF problem.

To solve the above SCF problem $ \rho = D(V(\rho)) $
one usually follows an iterative procedure.
That is to say that starting from an initial guess $\rho_0$ one then
computes $D(V(\rho_n))$ for a sequence of iterates $\rho_n$ until input and output
are close enough. That is until the residual
```math
R(\rho_n) = D(V(\rho_n)) - \rho_n
```
is small. Details on such SCF algorithms will be discussed in [Self-consistent field methods](@ref).
