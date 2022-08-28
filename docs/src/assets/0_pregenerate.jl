using MKL
using DFTK
setup_threading(n_blas=2)

let
    include("../../../../examples/convergence_study.jl")

    result = converge_kgrid(nkpts; Ecut=mean(Ecuts), tol)
    nkpt_conv = result.nkpt_conv
    p = plot(result.nkpts, result.errors, dpi=300, lw=3, m=:o, yaxis=:log,
             xlabel="k-grid", ylabel="energy absolute error", label="")
    savefig(p, "convergence_study_kgrid.png")

    result = converge_Ecut(Ecuts; nkpt=nkpt_conv, tol)
    p = plot(result.Ecuts, result.errors, dpi=300, lw=3, m=:o, yaxis=:log,
             xlabel="Ecut", ylabel="energy absolute error", label="")
    savefig(p, "convergence_study_ecut.png")
end
