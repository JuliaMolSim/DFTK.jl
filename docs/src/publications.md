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
  title = {{DFTK}: A {Julian} approach for simulating electrons in solids},
  volume = {3},
  pages = {69},
  year = {2021},
}
```

Additionally the following publications describe DFTK or one of its algorithms:

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


## Dependencies and third-party software
DFTK builds upon the work of many scientific libraries and computational tools.
We are grateful to the developers of these packages and encourage users to cite
the relevant papers when using DFTK in their research.

### Core numerical libraries

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

### Pseudopotentials and structure databases

- **PseudoPotentialData.jl**: Provides access to pseudopotential databases.
  Built upon [PseudoDojo](http://www.pseudo-dojo.org/) and other standard pseudopotential tables.
  M. J. van Setten, M. Giantomassi, E. Bousquet, M. J. Verstraete, D. R. Hamann, X. Gonze, and G.-M. Rignanese.
  [*The PseudoDojo: Training and grading an 85 element optimized norm-conserving pseudopotential table*](https://doi.org/10.1016/j.cpc.2018.01.012).
  Computer Physics Communications **226**, 39 (2018).

### Optimization and linear algebra

- **Optim.jl**: Optimization algorithms for Julia.
  P. K. Mogensen and A. N. Riseth.
  [*Optim: A mathematical optimization package for Julia*](https://doi.org/10.21105/joss.00615).
  Journal of Open Source Software **3**, 615 (2018).

- **KrylovKit.jl**: Krylov-based algorithms for linear problems, singular value and eigenvalue problems.
  [https://github.com/Jutho/KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl)

### Additional resources

For a complete list of DFTK's dependencies and their version information,
you can call `DFTK.versioninfo()` in your Julia session or examine
the `Project.toml` file in the DFTK repository.

## Research conducted with DFTK
The following publications report research employing DFTK as a core component.
Feel free to drop us a line if you want your work to be added here.

- X. Gonze, C. Tantardini, A. Levitt.
  [*Low-temperature behavior of density-functional theory for metals based on density-functional perturbation theory and Sommerfeld expansion*](https://doi.org/10.1103/yj83-j9p1) Physical Review B **113**, 035125 (2026). ([Computational script](https://github.com/antoine-levitt/temperature_perturbation_theory))

- X. Quan, H. Chen.
  [*Stochastic Density Functional Theory Through the Lens of Multilevel Monte Carlo Method*](https://arxiv.org/abs/2512.04860v2) (2025).

- D. Petersheim, J.-F. Pietschmann, J. Püschel, K. Ruess.
  [*Neural Network Acceleration of Iterative Methods for Nonlinear Schrödinger Eigenvalue Problems*](https://arxiv.org/abs/2507.16349) (2025).

- A. Levitt, D. Lundholm, N. Rougerie.
  [*Magnetic Thomas-Fermi theory for 2D abelian anyons*](https://arxiv.org/abs/2504.13481) (2025).

- D. Petersheim, J. Püschel, T. Stykel.
  [*Energy-adaptive Riemannian Conjugate Gradient method for density functional theory*](https://arxiv.org/abs/2503.16225) (2025).
  ([Implementation](https://github.com/jonas-pueschel/RCG_DFTK/)).

- A. Bordignon, G. Dusson, E. Cancès, G. Kemlin, R. A. L. Reyes and B. Stamm.
  [*Fully guaranteed and computable error bounds on the energy for periodic Kohn-Sham equations with convex density functionals*](http://arxiv.org/abs/2409.11769v1) (2024).
  ([Supplementary material and computational scripts](https://doi.org/10.18419/darus-4469)).

- M. F. Herbst, V. H. Bakkestuen, A. Laestadius.
  [*Kohn-Sham inversion with mathematical guarantees*](https://doi.org/10.1103/PhysRevB.111.205143)
  Phys. Rev. B, **111**, 205143 (2025).
  [ArXiv:2409.04372](https://arxiv.org/abs/2409.04372).
  ([Supplementary material and computational scripts](https://github.com/mfherbst/supporting-my-inversion)).

- J. Cazalis.
  [*Dirac cones for a mean-field model of graphene*](https://doi.org/10.2140/paa.2024.6.129)
  Pure and Appl. Anal., **6**, 1 (2024).
  [ArXiv:2207.09893](https://arxiv.org/abs/2207.09893).
  ([Computational script](https://github.com/JuliaMolSim/DFTK.jl/blob/f7fcc31c79436b2582ac1604d4ed8ac51a6fd3c8/examples/publications/2022_cazalis.jl)).

- E. Cancès, G. Kemlin, A. Levitt.
  [*A Priori Error Analysis of Linear and Nonlinear Periodic Schrödinger Equations with Analytic Potentials*](https://doi.org/10.1007/s10915-023-02421-0)
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
