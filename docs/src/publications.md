# Publications

Since DFTK is mostly developed as part of academic research,
we would greatly appreciate if you cite our research papers as appropriate.
See the [CITATION.bib](https://github.com/JuliaMolSim/DFTK.jl/blob/master/CITATION.bib)
in the root of the DFTK repo and the publication list
on this page for relevant citations. See also the [research page](research.md) for
research conducted with DFTK.

## DFTK reference paper

The current DFTK reference paper to cite is
```bibtex
@article{DFTKjcon,
  author = {Michael F. Herbst and Antoine Levitt and Eric Cancès},
  doi = {10.21105/jcon.00069},
  journal = {Proc. JuliaCon Conf.},
  title = {{DFTK}: A {Julian} approach for simulating electrons in solids},
  volume = {3},
  pages = {69},
  year = {2021},
}
```

## DFTK algorithms

The following publications describe DFTK algorithms:

- N. F. Schmitz, B. Ploumhans and M. F. Herbst.
  [*Algorithmic differentiation for plane-wave DFT: materials design, error control and learning model parameters.*](https://doi.org/10.1038/s41524-025-01880-3) npj Computational Materials **12**, 6 (2026).
  ([Supplementary material and computational scripts](https://github.com/niklasschmitz/ad-dfpt)).

- M. F. Herbst, B. Sun.
  [*Efficient Krylov methods for linear response in plane-wave electronic structure calculations.*](https://arxiv.org/abs/2505.02319) (2025).
  ([Supplementary material and computational scripts](https://github.com/bonans/inexact_Krylov_response)).

- E. Cancès, M. Hassan and L. Vidal.
  [*Modified-Operator Method for the Calculation of Band Diagrams of Crystalline Materials.*](https://doi.org/10.1090/mcom/3897)
  Math. Comp. **93**, 1203 (2024).
  [ArXiv:2210.00442](https://arxiv.org/abs/2210.00442).
  ([Supplementary material and computational scripts](https://github.com/LaurentVidal95/ModifiedOp)).

- E. Cancès, M. F. Herbst, G. Kemlin, A. Levitt and B. Stamm.
  [*Numerical stability and efficiency of response property calculations in density functional theory*](https://arxiv.org/abs/2210.04512)
  Letters in Mathematical Physics, **113**, 21 (2023).
  [ArXiv:2210.04512](https://arxiv.org/abs/2210.04512).
  ([Supplementary material and computational scripts](https://github.com/gkemlin/response-calculations-metals)).

- M. F. Herbst and A. Levitt.
  [*A robust and efficient line search for self-consistent field iterations*](https://arxiv.org/abs/2109.14018)
  Journal of Computational Physics, **459**, 111127 (2022).
  [ArXiv:2109.14018](https://arxiv.org/abs/2109.14018).
  ([Supplementary material and computational scripts](https://github.com/mfherbst/supporting-adaptive-damping/)).

- M. F. Herbst, A. Levitt and E. Cancès.
  [*DFTK: A Julian approach for simulating electrons in solids.*](https://doi.org/10.21105/jcon.00069)
  JuliaCon Proceedings, **3**, 69 (2021).

- M. F. Herbst and A. Levitt.
  [*Black-box inhomogeneous preconditioning for self-consistent field iterations in density functional theory*](https://doi.org/10.1088/1361-648X/abcbdb).
  Journal of Physics: Condensed Matter, **33**, 085503 (2021).
  [ArXiv:2009.01665](https://arxiv.org/abs/2009.01665).
  ([Supplementary material and computational scripts](https://github.com/mfherbst/supporting-ldos-preconditioning/)).

## Dependencies
DFTK builds upon the work of many scientific libraries and computational tools,
some of which are listed below. We are grateful to the developers of these packages
and encourage users to cite the relevant papers when using DFTK in their research.

- **Libxc**: Library of exchange-correlation functionals for density-functional theory.
  S. Lehtola, C. Steigemann, M. J. T. Oliveira, and M. A. L. Marques.
  [*Recent developments in libxc — A comprehensive library of functionals for density functional theory*](https://doi.org/10.1016/j.softx.2017.11.002).
  SoftwareX **7**, 1 (2018).
  ```bibtex
  @article{Lehtola2018,
    author = {Lehtola, Susi and Steigemann, Conrad and Oliveira, Micael J.T. and Marques, Miguel A.L.},
    title = {Recent developments in libxc --- A comprehensive library of functionals for density functional theory},
    journal = {SoftwareX},
    volume = {7},
    pages = {1--5},
    year = {2018},
    doi = {10.1016/j.softx.2017.11.002}
  }
  ```

- **FFTW**: Fastest Fourier Transform in the West.
  M. Frigo and S. G. Johnson.
  [*The design and implementation of FFTW3*](https://doi.org/10.1109/JPROC.2004.840301).
  Proceedings of the IEEE **93**, 216 (2005).
  ```bibtex
  @article{FFTW05,
    author = {Frigo, Matteo and Johnson, Steven G.},
    title = {The design and implementation of {FFTW3}},
    journal = {Proceedings of the IEEE},
    volume = {93},
    number = {2},
    pages = {216--231},
    year = {2005},
    doi = {10.1109/JPROC.2004.840301}
  }
  ```

- **Spglib**: Library for finding and handling crystal symmetries.
  A. Togo and I. Tanaka.
  [*Spglib: a software library for crystal symmetry search*](https://arxiv.org/abs/1808.01590).
  [ArXiv:1808.01590](https://arxiv.org/abs/1808.01590) (2018).
  ```bibtex
  @misc{Spglib,
    title = {{Spglib}: a software library for crystal symmetry search},
    author = {Atsushi Togo and Isao Tanaka},
    year = {2018},
    eprint = {1808.01590},
    archivePrefix = {arXiv},
    primaryClass = {cond-mat.mtrl-sci}
  }
  ```

- **Optim.jl**: Optimization algorithms for Julia.
  P. K. Mogensen and A. N. Riseth.
  [*Optim: A mathematical optimization package for Julia*](https://doi.org/10.21105/joss.00615).
  Journal of Open Source Software **3**, 615 (2018).

- **KrylovKit.jl**: Krylov-based algorithms for linear problems, singular value and eigenvalue problems.
  [https://github.com/Jutho/KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl)
