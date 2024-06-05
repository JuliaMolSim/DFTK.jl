# Publications

Since DFTK is mostly developed as part of academic research,
we would greatly appreciate if you cite our research papers as appropriate.
See the [CITATION.bib](https://github.com/JuliaMolSim/DFTK.jl/blob/master/CITATION.bib)
in the root of the DFTK repo and the publication list
on this page for relevant citations.
The current DFTK reference paper to cite is
```bibtex
@article{DFTKjcon,
  author = {Michael F. Herbst and Antoine Levitt and Eric Cancès},
  doi = {10.21105/jcon.00069},
  journal = {Proc. JuliaCon Conf.},
  title = {DFTK: A Julian approach for simulating electrons in solids},
  volume = {3},
  pages = {69},
  year = {2021},
}
```

Additionally the following publications describe DFTK or one of its algorithms:

- E. Cancès, M. Hassan and L. Vidal.
  [*Modified-Operator Method for the Calculation of Band Diagrams of Crystalline Materials.*](https://doi.org/10.1090/mcom/3897)
  Math. Comp. **93**, 1203 (2024).
  [ArXiv:2210.00442](https://arxiv.org/abs/2210.00442).
  ([Supplementary material and computational scripts](https://github.com/LaurentVidal95/ModifiedOp).

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


## Research conducted with DFTK
The following publications report research employing DFTK as a core component.
Feel free to drop us a line if you want your work to be added here.

- J. Cazalis.
  [*Dirac cones for a mean-field model of graphene*](https://doi.org/10.2140/paa.2024.6.129)
  Pure and Appl. Anal., **6**, 1 (2024).
  [ArXiv:2207.09893](https://arxiv.org/abs/2207.09893).
  ([Computational script](https://github.com/JuliaMolSim/DFTK.jl/blob/f7fcc31c79436b2582ac1604d4ed8ac51a6fd3c8/examples/publications/2022_cazalis.jl)).

- E. Cancès, G. Kemlin, A. Levitt.
  [*A Priori Error Analysis of Linear and Nonlinear Periodic Schr\"{o}dinger Equations with Analytic Potentials*](https://doi.org/10.1007/s10915-023-02421-0)
  J. Sci. Comp., **98**, 25 (2024).
  [ArXiv:2206.04954](https://arxiv.org/abs/2206.04954).

- E. Cancès, L. Garrigue, D. Gontier.
  [*A simple derivation of moiré-scale continuous models for twisted bilayer graphene*](https://doi.org/10.1103/PhysRevB.107.155403)
  Physical Review B, **107**, 155403 (2023).
  [ArXiv:2206.05685](https://arxiv.org/abs/2206.05685).

- G. Dusson, I. Sigal and B. Stamm.
  [*Analysis of the Feshbach-Schur method for the Fourier spectral discretizations of Schrödinger operators*](http://doi.org/10.1090/mcom/3774)
  Mathematics of Computation, **92**, 217 (2023).
  [ArXiv:2008.10871](https://arxiv.org/abs/2008.10871).

- E. Cancès, G. Dusson, G. Kemlin and A. Levitt.
  [*Practical error bounds for properties in plane-wave electronic structure calculations*](https://doi.org/10.1137/21M1456224)
  SIAM Journal on Scientific Computing, **44**, B1312 (2022).
  [ArXiv:2111.01470](https://arxiv.org/abs/2111.01470).
  ([Supplementary material and computational scripts](https://github.com/gkemlin/paper-forces-estimator)).

- E. Cancès, G. Kemlin and A. Levitt.
  [*Convergence analysis of direct minimization and self-consistent iterations*](https://doi.org/10.1137/20M1332864)
  SIAM Journal on Matrix Analysis and Applications, **42**, 243 (2021).
  [ArXiv:2004.09088](https://arxiv.org/abs/2004.09088).
  ([Computational script](https://github.com/JuliaMolSim/DFTK.jl/blob/80c7452ef728f5e9f413f70e6d5eb4f8357075bc/examples/silicon_scf_convergence.jl)).

- M. F. Herbst, A. Levitt and E. Cancès.
  [*A posteriori error estimation for the non-self-consistent Kohn-Sham equations.*](https://doi.org/10.1039/D0FD00048E)
  Faraday Discussions, **224**, 227 (2020).
  [ArXiv:2004.13549](https://arxiv.org/abs/2004.13549).
  ([Reference implementation](https://github.com/mfherbst/error-estimates-nonscf-kohn-sham)).
