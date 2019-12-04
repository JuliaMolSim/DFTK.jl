Self-consistent field iteration.

The SCF works in two stages: first, formulate a (hopefully well-conditioned) fixed-point problem (eg simple or Kerker mixing/preconditioner, see mixing.jl). Then, solve it using an acceleration method (eg Anderson, see scf_solvers.jl).
