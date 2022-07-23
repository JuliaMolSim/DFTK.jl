# DFTK.jl: The density-functional toolkit.

The density-functional toolkit, **DFTK** for short, is a library of
Julia routines for playing with plane-wave
density-functional theory (DFT) algorithms.
In its basic formulation it solves periodic Kohn-Sham equations.
The unique feature of the code is its **emphasis on simplicity
and flexibility**
with the goal of facilitating methodological development and
interdisciplinary collaboration.
In about 7k lines of pure Julia code
we support a [sizeable set of features](@ref package-features).
Our performance is of the same order of magnitude as much larger production
codes such as [Abinit](https://www.abinit.org/),
[Quantum Espresso](http://quantum-espresso.org/) and
[VASP](https://www.vasp.at/).
DFTK's source code is [publicly available on github](https://dftk.org).

If you are new to density-functional theory or plane-wave methods,
see our notes on [Periodic problems](@ref periodic-problems) and our
collection of [Introductory resources](@ref).

!!! tip "DFTK summer school: 29th to 31st August 2022 in Paris, France"
    We will organise a summer school centred around the DFTK code
    and modern approaches to density-functional theory
    from **29 to 31 August 2022** at **Sorbonne Université, Paris**.
    For more details and registration info see the [school's website](https://school2022.dftk.org).

# [Getting started](@id getting-started)
First, new users should take a look at the [Installation](@ref) and
[Tutorial](@ref) sections. Then, make your way through the various examples.
An ideal starting point are the [Examples on basic DFT calculations](@ref metallic-systems).

!!! note "Convergence parameters in the documentation"
    In the documentation we use very rough convergence parameters to be able
    to automatically generate this documentation very quickly.
    Therefore results are far from converged.
    Tighter thresholds and larger grids should be used for
    more realistic results.

If you have an idea for an addition to the docs or see something wrong,
please open an [issue](https://github.com/JuliaMolSim/DFTK.jl/issues)
or [pull request](https://github.com/JuliaMolSim/DFTK.jl/pulls)!
