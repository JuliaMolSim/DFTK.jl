using DFTK
using Optim
using HDF5
using TimerOutputs

using LinearAlgebra
using FFTW

BLAS.set_num_threads(4)
FFTW.set_num_threads(4)

include("perturbations.jl")

aref = 10.263141334305942
Eref = -7.924456699632 # computed with silicon.jl for kgrid = [4,4,4], Ecut = 100
tol_nrj = 1e-8
tol_a = eps()

# variable to store the planewaves and densities from one iteration to another
use_previous_iteration = true
ρ_start = nothing
ψ_start = nothing
prev_basis = nothing
count_fft = 0
count_ifft = 0

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

    # precize the number of electrons and build the model
    Ne = 8
    model = model_LDA(lattice, atoms; n_electrons=Ne)
    kcoords, ksymops = bzmesh_ir_wedge(kgrid, model.symops)
    basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops)

    global ρ_start, ψ_start, prev_basis
    if use_previous_iteration
        if ρ_start === nothing
            ρ_start = guess_density(basis)
        else
            ρ_start = DFTK.interpolate_density(ρ_start, basis)
        end
        if prev_basis === nothing || prev_basis === nothing
            ψ_start = nothing
        else
            ψ_start, _ = DFTK.interpolate_blochwave(ψ_start, prev_basis, basis)
        end
    end
    scfres = self_consistent_field(basis, tol=tol_nrj, ρ=ρ_start, ψ=ψ_start,
                                   callback=info->nothing)

    if use_previous_iteration
        ρ_start = scfres.ρ
        ψ_start = scfres.ψ
        prev_basis = scfres.ham.basis
    end
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
    if use_previous_iteration
        if ρ_start === nothing
            ρ_start = guess_density(basis)
        else
            ρ_start = DFTK.interpolate_density(ρ_start, basis)
        end
        if prev_basis === nothing || prev_basis === nothing
            ψ_start = nothing
        else
            ψ_start, _ = DFTK.interpolate_blochwave(ψ_start, prev_basis, basis)
        end
    end
    scfres = self_consistent_field(basis, tol=tol_nrj, ρ=ρ_start, ψ=ψ_start,
                                   callback=info->nothing)

    E_p, _ = perturbation(basis, kcoords, ksymops, scfres, α*Ecut;
                          compute_egval=false)

    if use_previous_iteration
        ρ_start = scfres.ρ
        ψ_start = scfres.ψ
        prev_basis = scfres.ham.basis
    end
    sum(values(E_p))
end


"""
optimize the lattice constant for a fixed Ecut
"""
function optimize_a(Ecut)

    function f(x)
        E(x, Ecut)
    end

    res = @timed Optim.minimizer(optimize(f, 0.95*aref, 1.05*aref,
                                          rel_tol=sqrt(tol_a), abs_tol=tol_a))
end

"""
optimize the lattice constant for a fixed Ecut with perturbation
"""
function optimize_a_perturbed(Ecut, α)

    function f(x)
        E_perturbed(x, Ecut, α)
    end

    res = @timed Optim.minimizer(optimize(f, 0.95*aref, 1.05*aref,
                                          rel_tol=sqrt(tol_a), abs_tol=tol_a))
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
    res_ref = optimize_a(90)

    Ecut_list = 10:5:80
    time_list = []
    alloc_list = []
    err_list = []
    a_list = []
    fft_list = []
    ifft_list = []

    Ecut_p_list = 10:5:15
    time_p_list = []
    alloc_p_list = []
    err_p_list = []
    a_p_list = []
    fft_p_list = []
    ifft_p_list = []

    for Ecut in Ecut_list

        println("------------------------------------------------------")
        println("Ecut = $(Ecut)")

        global ρ_start, ψ_start, prev_basis
        ρ_start = nothing
        ψ_start = nothing
        prev_basis = nothing
        global count_fft, count_ifft
        count_fft = 0
        count_ifft = 0
        reset_timer!(DFTK.timer)
        res = optimize_a(Ecut)
        display(DFTK.timer)
        ifft_num = TimerOutputs.ncalls(DFTK.timer["G_to_r!"])
        fft_num = TimerOutputs.ncalls(DFTK.timer["r_to_G!"])
        println("------------------------------------------------------")
        push!(a_list, res[1])
        push!(err_list, abs(res[1] - res_ref[1]))
        push!(time_list, res[2])
        push!(alloc_list, res[3])
        push!(ifft_list, ifft_num)
        push!(fft_list, fft_num)
    end

    for Ecut_p in Ecut_p_list

        println("------------------------------------------------------")
        println("Ecut_p = $(Ecut_p)")

        global ρ_start, ψ_start, prev_basis
        ρ_start = nothing
        ψ_start = nothing
        prev_basis = nothing
        global count_fft, count_ifft
        count_fft = 0
        count_ifft = 0
        reset_timer!(DFTK.timer)
        res_p = optimize_a_perturbed(Ecut_p, 2.5)
        display(DFTK.timer)
        ifft_p_num = TimerOutputs.ncalls(DFTK.timer["G_to_r!"])
        fft_p_num = TimerOutputs.ncalls(DFTK.timer["r_to_G!"])
        println("------------------------------------------------------")
        push!(a_p_list, res_p[1])
        push!(err_p_list, abs(res_p[1] - res_ref[1]))
        push!(time_p_list, res_p[2])
        push!(alloc_p_list, res_p[3])
        push!(ifft_p_list, ifft_p_num)
        push!(fft_p_list, fft_p_num)
    end

    h5open("optim_a_noperturb.h5", "w") do file
        file["aref"] = res_ref[1]
        file["Ecut_list"] = collect(Ecut_list)
        file["time_list"] = Float64.(time_list)
        file["err_list"]  = Float64.(err_list)
        file["a_list"]    = Float64.(a_list)
        file["fft_list"]  = Float64.(fft_list)
        file["ifft_list"]  = Float64.(ifft_list)
        file["Ecutp_list"] = Float64.(collect(Ecut_p_list))
        file["timep_list"] = Float64.(time_p_list)
        file["errp_list"]  = Float64.(err_p_list)
        file["ap_list"]    = Float64.(a_p_list)
        file["ifftp_list"] = Float64.(ifft_p_list)
        file["fftp_list"] = Float64.(fft_p_list)
    end

end
compare()
############################# Graphical ########################################

"""
graphical optimization of the lattice constant
"""
function optimize_a_graphic(Ecut)

    E_list = []
    E_p_list = []
    for a in arange
        println("Solving for a=$(a)")
        global ρ_start, ψ_start, prev_basis
        ρ_start = nothing
        ψ_start = nothing
        prev_basis = nothing
        push!(E_list, E(a, Ecut))
        global ρ_start, ψ_start, prev_basis
        ρ_start = nothing
        ψ_start = nothing
        prev_basis = nothing
        push!(E_p_list, E_perturbed(a, Ecut, 2.5))
    end
    plot(arange, E_list, "+-", label="Energy with Ecut=$(Ecut)")
    plot(arange, E_p_list, "x-", label="Perturbed energy with Ecut=$(Ecut)")

    h5open("a_optim_graphic.h5", "r+") do file
        file["Ecut_$(Ecut)/perturbed"] = Float64.(E_p_list)
        file["Ecut_$(Ecut)/full"] = Float64.(E_list)
    end
end

Ecut_list = 10:5:20
arange = range(0.95*aref, 1.05*aref, length=51)
h5open("a_optim_graphic.h5", "w") do file
    file["Ecut_list"] = collect(Ecut_list)
    file["arange"] = collect(arange)
end
for Ecut in Ecut_list
    optimize_a_graphic(Ecut)
end
