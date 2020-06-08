using DFTK
using PyPlot
using DataFrames
using GLM
import Statistics: mean
using Optim
using HDF5

include("perturbations.jl")

aref = 10.263141334305942
Eref = -7.924456699632 # computed with silicon.jl for kgrid = [4,4,4], Ecut = 100
tol = 1e-6

### Setting the model
# Calculation parameters
kgrid = [4, 4, 4]       # k-Point grid
supercell = [1, 1, 1]   # Lattice supercell

# Setup silicon lattice
lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

# Make a supercell if desired
pystruct = pymatgen_structure(lattice, atoms)
pystruct.make_supercell(supercell)
lattice = load_lattice(pystruct)
atoms = [Si => [s.frac_coords for s in pystruct.sites]]

# precize the number of electrons on build the model
Ne = 8
model = model_LDA(lattice, atoms; n_electrons=Ne)
kcoords, ksymops = bzmesh_ir_wedge(kgrid, model.symops)
basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops)

# variable to store the plane waves from one iteration to another
ψ = nothing

"""
compute scfres for a given lattice constant
"""
DFTK.@timing function E(a, Ecut)

    scfres = self_consistent_field(basis, tol=tol/10, ψ=ψ,
                                   callback=info->nothing)

    global ψ = scfres.ψ
    sum(values(scfres.energies))
end

"""
compute scfres with perturbation coef α
"""
DFTK.@timing function E_perturbed(a, Ecut, α)

    scfres = self_consistent_field(basis, tol=tol/10, ψ=ψ,
                                   callback=info->nothing)

    global ψ = scfres.ψ
    avg = true
    Ep_fine, _ = perturbation(basis, kcoords, ksymops, scfres, α*Ecut, false)
    sum(values(Ep_fine))
end


"""
optimize the lattice constant for a fixed Ecut
"""
DFTK.@timing function optimize_a(Ecut)

    function f(x)
        E(x, Ecut)
    end

    res = @timed Optim.minimizer(optimize(f, 0.95*aref, 1.05*aref, GoldenSection()))
    #  amin = Optim.minimizer(res)
end

"""
optimize the lattice constant for a fixed Ecut with perturbation
"""
function optimize_a_perturbed(Ecut, α)

    function f(x)
        E_perturbed(x, Ecut, α)
    end

    res = @timed Optim.minimizer(optimize(f, 0.95*aref, 1.05*aref, GoldenSection()))
    #  amin = Optim.minimizer(res)
end

"""
compare optimization time for non-perturbed and perturbed scf
"""
function compare()

    global ψ = nothing
    res_ref = optimize_a(40)

    Ecut_list = 10:5:15
    time_list = []
    alloc_list = []
    err_list = []
    a_list = []

    Ecutp_list = 10:5:15
    timep_list = []
    allocp_list = []
    errp_list = []
    ap_list = []

    for Ecut in Ecut_list

        println("----------------------")
        println("Ecut = $(Ecut)")
        global ψ = nothing

        res = optimize_a(Ecut)
        push!(a_list, res[1])
        push!(err_list, abs(res[1] - res_ref[1]))
        push!(time_list, res[2])
        push!(alloc_list, res[3])
    end

    for Ecutp in Ecutp_list

        println("----------------------")
        println("Ecutp = $(Ecutp)")
        global ψ = nothing

        resp = optimize_a_perturbed(Ecutp, 2.5)
        push!(ap_list, resp[1])
        push!(errp_list, abs(resp[1] - res_ref[1]))
        push!(timep_list, resp[2])
        push!(allocp_list, resp[3])
    end

    h5open("optim_a.h5", "w") do file
        file["aref"] = res_ref[1]
        file["Ecut_list"] = collect(Ecut_list)
        file["time_list"] = Float64.(time_list)
        file["err_list"]  = Float64.(err_list)
        file["a_list"]    = Float64.(a_list)
        file["Ecutp_list"] = Float64.(collect(Ecutp_list))
        file["timep_list"] = Float64.(timep_list)
        file["errp_list"]  = Float64.(errp_list)
        file["ap_list"]    = Float64.(ap_list)
    end

end
reset_timer!(DFTK.timer)
compare()
display(DFTK.timer)
STOP
############################# Graphical ########################################

"""
graphical optimization of the lattice constant
"""
function optimize_a_graphic(Ecut)
    arange = range(0.95*aref, 1.05*aref, length=11)

    E_list = []
    Ep_list = []
    for a in arange
        println("Solving for a=$(a)")
        push!(E_list, E(a, Ecut))
        push!(Ep_list, E_perturbed(a, Ecut, 2.5))
    end
    PyPlot.plot(arange, E_list, "+-", label="Energy with Ecut=$(Ecut)")
    PyPlot.plot(arange, Ep_list, "o-", label="Perturbed energy with Ecut=$(Ecut)")
end

#  for Ecut in 10:5:20
#      println("--------------------------------")
#      println("Ecut=$(Ecut)")
#      amin = optimize_a_graphic(Ecut)
#      println(amin)
#  end
#  legend()






