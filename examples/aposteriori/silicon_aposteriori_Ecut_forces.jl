# Very basic setup, useful for testing
using DFTK
using PyPlot

include("aposteriori_forces.jl")
include("aposteriori_tools.jl")
include("aposteriori_callback.jl")

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8 .+ 0.02, -ones(3)/8]]

model = Model(lattice; atoms=atoms, n_electrons=2,
              terms=[Kinetic(), AtomicLocal()])
nl = false
#  model = model_LDA(lattice, atoms; n_electrons=2)
kgrid = [1,1,1]  # k-point grid (Regular Monkhorst-Pack grid)
Ecut_ref = 50           # kinetic energy cutoff in Hartree
tol = 1e-10
tol_krylov = 1e-12
basis_ref = PlaneWaveBasis(model, Ecut_ref; kgrid=kgrid)

filled_occ = DFTK.filled_occupation(model)
N = div(model.n_electrons, filled_occ)
Nk = length(basis_ref.kpoints)
T = eltype(basis_ref)
occupation = [filled_occ * ones(T, N) for ik = 1:Nk]

scfres_ref = self_consistent_field(basis_ref, tol=tol,
                                   determine_diagtol=DFTK.ScfDiagtol(diagtol_max=1e-10),
                                   is_converged=DFTK.ScfConvergenceDensity(tol))

f_ref = norm(forces(scfres_ref)[1][1])

Ecut_list = 10:5:Ecut_ref-5
forces_list = []
estimator_forces_list = []

for Ecut in Ecut_list

    println("--------------------------")
    println("Ecut = $(Ecut)")

    basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
    scfres = self_consistent_field(basis, tol=tol,
                                   determine_diagtol=DFTK.ScfDiagtol(diagtol_max=1e-10),
                                   is_converged=DFTK.ScfConvergenceDensity(tol))

    ## explicit computation of f-f*
    f = norm(forces(scfres)[1][1])
    append!(forces_list, abs(f-f_ref))

    ## f_est ~ |f-f*|
    # compute residual
    φ = similar(scfres.ψ)
    for ik = 1:Nk
        φ[ik] = scfres.ψ[ik][:,1:N]
    end
    φr = DFTK.interpolate_blochwave(φ, basis, basis_ref)
    res = compute_residual(basis_ref, φr, occupation)

    f_est = norm(compute_forces_estimate(basis_ref, res, φr)[1][1])
    append!(estimator_forces_list, abs(f_est))
end

semilogy(Ecut_list, forces_list, label = "|F - F*|")
semilogy(Ecut_list, estimator_forces_list, label = "estimator")
xlabel("Ecut")
legend()

h5open("silicon_Ecut_lin_forces.h5", "w") do file
    file["Ecut_list"] = collect(Ecut_list)
    file["Ecut_ref"] = Ecut_ref
    file["kgrid"] = kgrid
    file["forces_list"] = Float64.(forces_list)
    file["estimator_forces_list"] = Float64.(estimator_forces_list)
    file["N"] = N
    file["nl"] = nl
end



