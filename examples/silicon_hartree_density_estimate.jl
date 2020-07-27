# Very basic setup, useful for testing
using DFTK
using PyPlot
using LinearAlgebra

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

model = model_atomic(lattice, atoms; extra_terms=[Hartree()])
#  model = model_LDA(lattice, atoms)
kgrid = [1, 1, 1]   # k-point grid (Regular Monkhorst-Pack grid)
Ecut = 15           # kinetic energy cutoff in Hartree
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
tol = 1e-10

# callback functions
default_callback = DFTK.ScfDefaultCallback()

density_differences = []
Vcdensity_differences = []
ρin_list = []
function plot_density_callback(info)
    if info.stage == :finalize
        ρ = info.ρ

        figure()
        semilogy(density_differences, "bx--", label="|ρout - ρin|")
        density_errors = [norm(ρin.real - ρ.real) for ρin in ρin_list]
        semilogy(density_errors, "rx--", label="|ρin - ρ*|")
        legend()

        semilogy(sqrt.(Vcdensity_differences), "bx-", label="√<ρout - ρin, Vc(ρout - ρin)>")
        Vcdensity_errors = []
        for ρin in ρin_list
            dρ = ρin - ρ
            Vcdρ = apply_kernel(basis.terms[6], dρ)
            push!(Vcdensity_errors, dot(dρ.real, Vcdρ.real))
        end
        semilogy(sqrt.(Vcdensity_errors), "rx-", label="√<ρin - ρ*, Vc(ρin - ρ*)>")
        legend()
    else
        default_callback(info)
        ρout = info.ρout
        ρin = info.ρin
        dρ = ρout - ρin
        Vcdρ = apply_kernel(basis.terms[6], dρ)
        push!(ρin_list, deepcopy(ρin))
        push!(Vcdensity_differences, dot(dρ.real, Vcdρ.real))
        push!(density_differences, norm(dρ.real))
    end
end

scfres = self_consistent_field(basis; tol=tol,
                               is_converged=DFTK.ScfConvergenceDensity(tol),
                               callback=plot_density_callback)
scfres.energies
