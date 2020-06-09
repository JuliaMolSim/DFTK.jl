using DFTK
using Optim
using HDF5
using TimerOutputs

using LinearAlgebra
using FFTW

BLAS.set_num_threads(16)
FFTW.set_num_threads(16)

include("perturbations.jl")

aref = 10.263141334305942
Eref = -7.924456699632 # computed with silicon.jl for kgrid = [4,4,4], Ecut = 100
tol = 1e-8
avg = true

# variable to store the plane waves from one iteration to another
ρ_start = nothing
ψ_start = nothing
prev_basis = nothing

"""
compute scfres for a given lattice constant
"""
DFTK.@timing function E(a, Ecut)
    ### Setting the model
    # Calculation parameters
    kgrid = [4, 4, 4]       # k-Point grid
    supercell = [1, 1, 1]   # Lattice supercell

    # Setup silicon lattice
    lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
    atoms = [Si => [ones(3)/8, -ones(3)/8]]

    # precize the number of electrons on build the model
    Ne = 8
    model = model_LDA(lattice, atoms; n_electrons=Ne)
    kcoords, ksymops = bzmesh_ir_wedge(kgrid, model.symops)
    basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops)

    global ρ_start, ψ_start, prev_basis
    if ρ_start === nothing
        ρ_start = guess_density(basis)
    else
        ρ_start = DFTK.interpolate_density(ρ_start, basis)
    end
    if prev_basis === nothing
        ψ_start = nothing
    else
        ψ_start, _ = DFTK.interpolate_blochwave(ψ_start, prev_basis, basis)
    end
    scfres = self_consistent_field(basis, tol=tol,
                                   ρ=ρ_start,
                                   ψ=ψ_start,
                                   callback=info->nothing)

    ρ_start = scfres.ρ
    ψ_start = scfres.ψ
    prev_basis = scfres.ham.basis
    sum(values(scfres.energies))
end

"""
compute scfres with perturbation coef α
"""
DFTK.@timing function E_perturbed(a, Ecut, α)
    ### Setting the model
    # Calculation parameters
    kgrid = [4, 4, 4]       # k-Point grid
    supercell = [1, 1, 1]   # Lattice supercell

    # Setup silicon lattice
    lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
    atoms = [Si => [ones(3)/8, -ones(3)/8]]

    # precize the number of electrons on build the model
    Ne = 8
    model = model_LDA(lattice, atoms; n_electrons=Ne)
    kcoords, ksymops = bzmesh_ir_wedge(kgrid, model.symops)
    basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops)

    global ρ_start, ψ_start, prev_basis
    if ρ_start === nothing
        ρ_start = guess_density(basis)
    else
        ρ_start = DFTK.interpolate_density(ρ_start, basis)
    end
    if prev_basis === nothing
        ψ_start = nothing
    else
        ψ_start, _ = DFTK.interpolate_blochwave(ψ_start, prev_basis, basis)
    end
    scfres = self_consistent_field(basis, tol=tol,
                                   ρ=ρ_start,
                                   ψ=ψ_start,
                                   callback=info->nothing)

    avg = true
    Ep_fine, _ = perturbation(basis, kcoords, ksymops, scfres, α*Ecut, false)

    ρ_start = scfres.ρ
    ψ_start = scfres.ψ
    prev_basis = scfres.ham.basis
    sum(values(Ep_fine))
end


"""
optimize the lattice constant for a fixed Ecut
"""
function optimize_a(Ecut)

    function f(x)
        E(x, Ecut)
    end

    res = @timed Optim.minimizer(optimize(f, 0.95*aref, 1.05*aref))
    #  amin = Optim.minimizer(res)
end

"""
optimize the lattice constant for a fixed Ecut with perturbation
"""
function optimize_a_perturbed(Ecut, α)

    function f(x)
        E_perturbed(x, Ecut, α)
    end

    res = @timed Optim.minimizer(optimize(f, 0.95*aref, 1.05*aref))
    #  amin = Optim.minimizer(res)
end

"""
compare optimization time for non-perturbed and perturbed scf
"""
function compare()

    # variable to store the plane waves from one iteration to another
    global ρ_start, ψ_start, prev_basis
    ρ_start = nothing
    ψ_start = nothing
    prev_basis = nothing
    res_ref = optimize_a(30)

    Ecut_list = 10:5:15
    time_list = []
    alloc_list = []
    err_list = []
    a_list = []
    fft_list = []
    ifft_list = []

    Ecutp_list = 10:5:15
    timep_list = []
    allocp_list = []
    errp_list = []
    ap_list = []
    fftp_list = []
    ifftp_list = []

    for Ecut in Ecut_list

        println("------------------------------------------------------")
        println("Ecut = $(Ecut)")

        global ρ_start, ψ_start, prev_basis
        ρ_start = nothing
        ψ_start = nothing
        prev_basis = nothing
        reset_timer!(DFTK.timer)
        res = optimize_a(Ecut)
        display(DFTK.timer)
        ifft_num = TimerOutputs.ncalls(DFTK.timer["E"]["self_consistent_field"]["LOBPCG"]["Hamiltonian multiplication"]["G_to_r!"]) + TimerOutputs.ncalls(DFTK.timer["E"]["self_consistent_field"]["compute_density"]["G_to_r!"]) + TimerOutputs.ncalls(DFTK.timer["E"]["self_consistent_field"]["energy_hamiltonian"]["G_to_r!"])

        fft_num = TimerOutputs.ncalls(DFTK.timer["E"]["self_consistent_field"]["LOBPCG"]["Hamiltonian multiplication"]["r_to_G!"]) + TimerOutputs.ncalls(DFTK.timer["E"]["self_consistent_field"]["compute_density"]["r_to_G!"]) + TimerOutputs.ncalls(DFTK.timer["E"]["self_consistent_field"]["energy_hamiltonian"]["r_to_G!"])


        println("------------------------------------------------------")
        push!(a_list, res[1])
        push!(err_list, abs(res[1] - res_ref[1]))
        push!(time_list, res[2])
        push!(alloc_list, res[3])
        push!(ifft_list, ifft_num)
        push!(fft_list, fft_num)
    end

    for Ecutp in Ecutp_list

        println("------------------------------------------------------")
        println("Ecutp = $(Ecutp)")

        global ρ_start, ψ_start, prev_basis
        ρ_start = nothing
        ψ_start = nothing
        prev_basis = nothing
        reset_timer!(DFTK.timer)
        resp = optimize_a_perturbed(Ecutp, 2.5)
        display(DFTK.timer)
        ifftp_num = TimerOutputs.ncalls(DFTK.timer["E_perturbed"]["self_consistent_field"]["LOBPCG"]["Hamiltonian multiplication"]["G_to_r!"]) + TimerOutputs.ncalls(DFTK.timer["E_perturbed"]["self_consistent_field"]["compute_density"]["G_to_r!"]) + TimerOutputs.ncalls(DFTK.timer["E_perturbed"]["self_consistent_field"]["energy_hamiltonian"]["G_to_r!"]) + TimerOutputs.ncalls(DFTK.timer["E_perturbed"]["perturbation"]["compute_density"]["G_to_r!"]) + TimerOutputs.ncalls(DFTK.timer["E_perturbed"]["perturbation"]["perturbed_eigenvalues"]["Hamiltonian multiplication"]["G_to_r!"]) + TimerOutputs.ncalls(DFTK.timer["E_perturbed"]["perturbation"]["perturbed_eigenvectors"]["Hamiltonian multiplication"]["G_to_r!"]) + TimerOutputs.ncalls(DFTK.timer["E_perturbed"]["perturbation"]["Rayleigh_Ritz"]["Hamiltonian multiplication"]["G_to_r!"]) + TimerOutputs.ncalls(DFTK.timer["E_perturbed"]["perturbation"]["energy_hamiltonian"]["G_to_r!"])

        fftp_num = TimerOutputs.ncalls(DFTK.timer["E_perturbed"]["self_consistent_field"]["LOBPCG"]["Hamiltonian multiplication"]["r_to_G!"]) + TimerOutputs.ncalls(DFTK.timer["E_perturbed"]["self_consistent_field"]["compute_density"]["r_to_G!"]) + TimerOutputs.ncalls(DFTK.timer["E_perturbed"]["self_consistent_field"]["energy_hamiltonian"]["r_to_G!"]) + TimerOutputs.ncalls(DFTK.timer["E_perturbed"]["perturbation"]["compute_density"]["r_to_G!"]) + TimerOutputs.ncalls(DFTK.timer["E_perturbed"]["perturbation"]["perturbed_eigenvalues"]["Hamiltonian multiplication"]["r_to_G!"]) + TimerOutputs.ncalls(DFTK.timer["E_perturbed"]["perturbation"]["perturbed_eigenvectors"]["Hamiltonian multiplication"]["r_to_G!"]) + TimerOutputs.ncalls(DFTK.timer["E_perturbed"]["perturbation"]["Rayleigh_Ritz"]["Hamiltonian multiplication"]["r_to_G!"])

        println("------------------------------------------------------")
        push!(ap_list, resp[1])
        push!(errp_list, abs(resp[1] - res_ref[1]))
        push!(timep_list, resp[2])
        push!(allocp_list, resp[3])
        push!(ifftp_list, ifftp_num)
        push!(fftp_list, fftp_num)
    end

    h5open("optim_a.h5", "w") do file
        file["aref"] = res_ref[1]
        file["Ecut_list"] = collect(Ecut_list)
        file["time_list"] = Float64.(time_list)
        file["err_list"]  = Float64.(err_list)
        file["a_list"]    = Float64.(a_list)
        file["fft_list"]  = Float64.(fft_list)
        file["ifft_list"]  = Float64.(ifft_list)
        file["Ecutp_list"] = Float64.(collect(Ecutp_list))
        file["timep_list"] = Float64.(timep_list)
        file["errp_list"]  = Float64.(errp_list)
        file["ap_list"]    = Float64.(ap_list)
        file["ifftp_list"] = Float64.(ifftp_list)
        file["fftp_list"] = Float64.(fftp_list)
    end

end
compare()
STOP
############################# Graphical ########################################

"""
graphical optimization of the lattice constant
"""
function optimize_a_graphic(Ecut)

    E_list = []
    Ep_list = []
    for a in arange
        println("Solving for a=$(a)")
        push!(E_list, E(a, Ecut))
        push!(Ep_list, E_perturbed(a, Ecut, 2.5))
    end
    #  plot(arange, E_list, "+-", label="Energy with Ecut=$(Ecut)")
    #  plot(arange, Ep_list, "o-", label="Perturbed energy with Ecut=$(Ecut)")

    h5open("a_optim_graphic.h5", "r+") do file
        file["Ecut_$(Ecut)/perturbed"] = Float64.(Ep_list)
        file["Ecut_$(Ecut)/full"] = Float64.(E_list)
    end
end

Ecut_list = 5:5:15
arange = range(0.95*aref, 1.05*aref, length=21)
h5open("a_optim_graphic.h5", "w") do file
    file["Ecut_list"] = collect(Ecut_list)
    file["arange"] = arange
end
for Ecut in Ecut_list
    optimize_a_graphic(Ecut)
end




