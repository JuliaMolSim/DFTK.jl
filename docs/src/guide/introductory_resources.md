# [Introductory resources](@id introductory-resources)

This page collects a bunch of articles, lecture notes, textbooks and recordings
related to density-functional theory (DFT) and DFTK.
Most introductory aspects of the code and the theory behind it are also covered
in the "Getting started" section of the documentation, e.g.
[Periodic problems and plane-wave discretisations](@ref periodic-problems),
[Introduction to density-functional theory](@ref)
or [Self-consistent field methods](@ref).

Since DFTK aims for an interdisciplinary audience the
level and scope of the referenced works varies.
They are roughly ordered from beginner to advanced.
For a list of articles dealing with novel research aspects achieved using DFTK,
see [Publications](@ref).

## Workshop material and tutorials
- [DFTK school 2022: Numerical methods for density-functional theory simulations](https://school2022.dftk.org):
  Summer school centred around the DFTK code and modern approaches to density-functional theory.
  See the [programme and lecture notes](https://school2022.dftk.org), in particular:
    * [Introduction to DFT](https://school2022.dftk.org/assets/Fromager_DFT.pdf)
    * [Introduction to periodic problems](https://school2022.dftk.org/assets/Bruneval_Solid_State_Planewave.pdf)
    * [Analysis of plane-wave DFT](https://school2022.dftk.org/assets/Cances_PlaneWave_DFT.pdf)
    * [Analysis of SCF convergence](https://github.com/mfherbst/dftk-workshop-material/tree/master/Lectures_Day2_Michael_Herbst)
    * [Exercises](https://github.com/mfherbst/dftk-workshop-material/tree/master/Exercises)


- [DFT in a nutshell](https://doi.org/10.1002/qua.24259) by Kieron Burke and Lucas Wagner:
  Short tutorial-style article introducing the basic DFT setting, basic equations and terminology.
  Great introduction from the physical / chemical perspective.

- [Workshop on mathematics and numerics of DFT](https://michael-herbst.com/teaching/2022-mit-workshop-dftk/):
  Two-day workshop at MIT centred around DFTK by M. F. Herbst,
  in particular the [summary of DFT theory](https://michael-herbst.com/teaching/2022-mit-workshop-dftk/2022-mit-workshop-dftk/DFT_Theory.pdf).

## Textbooks

- [Density Functional Theory](https://doi.org/10.1007/978-3-031-22340-2)
  edited by Eric Cancès and Gero Friesecke (Springer, 2023):
  Up to date textbook accessible to an interdisciplinary audience with contributions
  by mathematicians and application researchers. Particularly relevant are:
    * [Chapter 1: Review of Approximations for the Exchange-Correlation Energy in Density-Functional Theory](http://arxiv.org/abs/2103.02645v1)
    * [Chapter 7: Numerical Methods for Kohn–Sham Models: Discretization, Algorithms, and Error Analysis](https://doi.org/10.1007/978-3-031-22340-2_7)

- [Electronic Structure: Basic theory and practical methods](https://doi.org/10.1017/CBO9780511805769)
  by R. M. Martin (Cambridge University Press, 2004):
  Standard textbook introducing
  most common methods of the field (lattices, pseudopotentials, DFT, ...)
  from the perspective of a physicist.

- [A Mathematical Introduction to Electronic Structure Theory](http://dx.doi.org/10.1137/1.9781611975802)
  by L. Lin and J. Lu (SIAM, 2019):
  Monograph attacking DFT from a mathematical angle.
  Covers topics such as DFT, pseudos, SCF, response, ...

## Recordings
- [Julia for Materials Modelling](https://www.youtube.com/watch?v=dujepKxxxkg) by M. F. Herbst:
  One-hour talk providing an overview of materials modelling tools for Julia.
  Key features of DFTK are highlighted as part of the talk.
  [Pluto notebooks](https://mfherbst.github.io/julia-for-materials/)

- [DFTK: A Julian approach for simulating electrons in solids](https://www.youtube.com/watch?v=-RomkxjlIcQ) by M. F. Herbst:
  Pre-recorded talk for JuliaCon 2020.
  Assumes no knowledge about DFT and gives the broad picture of DFTK.
  [Slides](https://michael-herbst.com/talks/2020.07.29_juliacon_dftk.pdf).

- [Juliacon 2021 DFT workshop](https://www.youtube.com/watch?v=HvpPMWVm8aw) by M. F. Herbst:
  Three-hour workshop session at the 2021 Juliacon providing a mathematical look on
  DFT, SCF solution algorithms as well as the integration of DFTK into the Julia
  package ecosystem. Material starts to become a little outdated.
  [Workshop material](https://github.com/mfherbst/juliacon_dft_workshop)
