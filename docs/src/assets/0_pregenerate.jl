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

let
    include("../../../../examples/pseudopotentials.jl")

    function run_scf(Ecut, psp, tol)
        println("Ecut = $Ecut")
        println("----------------------------------------------------")
        a = 5.0
        lattice   = a * Matrix(I, 3, 3)
        atoms     = [ElementPsp(:Li, psp=psp)]
        positions = [zeros(3)]
    
        model = model_LDA(lattice, atoms, positions; temperature=1e-2)
        basis = PlaneWaveBasis(model; Ecut=Ecut, kgrid=[8, 8, 8])
        self_consistent_field(basis; tol)
    end
    
    function converge_Ecut(Ecuts, psp, tol)
        energies = [run_scf(Ecut, psp, tol/100).energies.total for Ecut in Ecuts]
        errors = abs.(energies[begin:end-1] .- energies[end])
        iconv = findfirst(errors .< tol)
        (Ecuts=Ecuts[begin:end-1], errors, Ecut_conv=Ecuts[iconv])
    end

    Ecuts = 20:4:96
    tol   = 1e-3

    conv_hgh = converge_Ecut(Ecuts, psp_hgh, tol)
    conv_upf = converge_Ecut(Ecuts, psp_upf, tol)

    println("HGH: $(onv_hgh.Ecut_conv)")
    println("UPF: $(conv_upf.Ecut_conv)")

    plt = plot(xlabel="Ecut [Ha]", ylabel="Error [Ha]")
    plot!(plt, conv_hgh.Ecuts, conv_hgh.errors, label="HGH")
    plot!(plt, conv_upf.Ecuts, conv_upf.errors, label="PseudoDojo UPF")
    hline!(plt, [tol], label="tol", color=:grey, linestyle=:dash)
    savefig(plt, "li_pseudos_ecut_convergence.png")
end