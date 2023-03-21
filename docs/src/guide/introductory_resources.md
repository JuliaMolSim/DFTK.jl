# [Introductory resources](@id introductory-resources)

This page collects a bunch of articles, lecture notes, textbooks and recordings
related to density-functional theory (DFT) and DFTK.
Since DFTK aims for an interdisciplinary audience the
level and scope of the referenced works varies.
They are roughly ordered from beginner to advanced.
For a list of articles dealing with novel research aspects achieved using DFTK,
see [Publications](@ref).

## Workshop material and tutorials
- [DFTK school 2022: Numerical methods for density-functional theory simulations](https://school2022.dftk.org):
  Summer school centred around the DFTK code and modern approaches to density-functional theory.
  [Programme and lecture notes](https://school2022.dftk.org), in particular:
    * [Introduction to DFT](https://school2022.dftk.org/assets/Fromager_DFT.pdf)
    * [Introduction to periodic problems](https://school2022.dftk.org/assets/Bruneval_Solid_State_Planewave.pdf)
    * [Analysis of plane-wave DFT](https://school2022.dftk.org/assets/Cances_PlaneWave_DFT.pdf)
    * [Analysis of SCF convergence](https://github.com/mfherbst/dftk-workshop-material/tree/master/Lectures_Day2_Michael_Herbst)
    * [Exercises](https://github.com/mfherbst/dftk-workshop-material/tree/master/Exercises)

- [DFT in a nutshell](https://doi.org/10.1002/qua.24259) by Kieron Burke and Lucas Wagner:
  Short tutorial-style article introducing the basic DFT setting, basic equations and terminology.
  Great introduction from the physical / chemical perspective.

- [Workshop on mathematics and numerics of DFT](https://michael-herbst.com/teaching/2022-mit-workshop-dftk/):
  Two-day workshop at MIT centred around DFTK by M. F. Herbst, in particular:
    * [Summary of DFT theory](https://michael-herbst.com/teaching/2022-mit-workshop-dftk/2022-mit-workshop-dftk/DFT_Theory.pdf)

## Textbooks
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
- [DFTK: A Julian approach for simulating electrons in solids](https://www.youtube.com/watch?v=-RomkxjlIcQ) by M. F. Herbst:
  Pre-recorded talk for JuliaCon 2020.
  Assumes no knowledge about DFT and gives the broad picture of DFTK.
  [Slides](https://michael-herbst.com/talks/2020.07.29_juliacon_dftk.pdf).

- [Juliacon 2021 DFT workshop](https://github.com/mfherbst/juliacon_dft_workshop) by M. F. Herbst:
  Three-hour workshop session at the 2021 Juliacon providing a mathematical look on
  DFT, SCF solution algorithms as well as the integration of DFTK into the Julia
  package ecosystem. Material starts to become a little outdated.
  [Youtube recording](https://www.youtube.com/watch?v=HvpPMWVm8aw).
