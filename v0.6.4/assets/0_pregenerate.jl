using MKL
using DFTK
using LinearAlgebra
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

    function run_scf(Ecut, psp)
        println("Ecut = $Ecut")
        println("----------------------------------------------------")
        a = 5.0
        lattice   = a * Matrix(I, 3, 3)
        atoms     = [ElementPsp(:Li; psp)]
        positions = [zeros(3)]

        model = model_LDA(lattice, atoms, positions; temperature=1e-2)
        basis = PlaneWaveBasis(model; Ecut, kgrid=[8, 8, 8])
        self_consistent_field(basis; tol=1e-8)
    end

    function converge_Ecut(Ecuts, psp, tol)
        energy_ref = run_scf(Ecuts[end], psp).energies.total
        energies = []
        for Ecut in Ecuts
            energy = run_scf(Ecut, psp).energies.total
            push!(energies, energy)
            if abs(energy - energy_ref) < tol
                break
            end
        end
        n_energies = length(energies)
        errors = abs.(energies .- energy_ref)
        iconv = findfirst(errors .< tol)
        (; Ecuts=Ecuts[begin:n_energies], errors,
         Ecut_conv=Ecuts[iconv], error_conv=errors[iconv])
    end

    n_atoms    = 1
    Ecuts      = 20:4:140
    tol_mev_at = 1.0u"meV" / n_atoms
    tol        = austrip(tol_mev_at)

    conv_upf = converge_Ecut(Ecuts, psp_upf, tol)
    println("UPF: $(conv_upf.Ecut_conv)")

    conv_hgh = converge_Ecut(Ecuts, psp_hgh, tol)
    println("HGH: $(conv_hgh.Ecut_conv)")

    plt = plot(yaxis=:log10, xlabel="Ecut [Eh]", ylabel="Error [Eh]")
    plot!(plt, conv_hgh.Ecuts, conv_hgh.errors, label="HGH",
          markers=true, linewidth=3)
    plot!(plt, conv_upf.Ecuts, conv_upf.errors, label="PseudoDojo UPF",
          markers=true, linewidth=3)
    hline!(plt, [tol], label="tol = $(round(typeof(1u"meV"), tol_mev_at, digits=3)) / atom",
           color=:grey, linestyle=:dash)
    scatter!(plt, [conv_hgh.Ecut_conv, conv_upf.Ecut_conv],
             [conv_hgh.error_conv, conv_upf.error_conv],
             color=:grey, label="", markers=:star, markersize=7)
    yticks!(plt, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
    savefig(plt, "li_pseudos_ecut_convergence.png")
end
