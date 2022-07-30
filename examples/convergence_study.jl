# # Performing a convergence study
#
# This example shows how to perform a convergence study to find the 
# appropriate kgrid and Ecut for a desired convergence and error tolerance.
#
# Generally, to wisely vary kgrid and Ecut, one can perform a convergence study by
# starting with a reasonable grid or Ecut, picking a property 
# such as energy, and increasing the grid
# until the property converges – its value does not change much. 

# This study must be performed separately for each value you wish to study (kgrid, Ecut, and even temperature).
# In this example we will use a Pt system to perform a convergence study for kgrid and Ecut in DFTK.
#
# We start by setting up a simple cubic Pt system

using DFTK
using Plots
using Unitful
using UnitfulAtomic
using LinearAlgebra
using Optim

nkpts = 1               # k-point grid
tol = 1e-2              # tolerance for the optimization routine
Ecut = 490u"eV"
temperature = 0.01
Pt = ElementPsp(:Pt, psp = load_psp("hgh/lda/Pt-q10"));
atoms = [Pt]
position = [zeros(3)]
a = 5.00 # in Bohr
lattice = a * I(3)

# Then we define a function to compute the total energy of the system for varying values of `Ecut` and `nkpts`
function compute_convergence(Ecut = 500, nkpts::Integer = 8, tol = 1e-2)
    model = model_LDA(lattice, atoms, position; temperature)
    basis = PlaneWaveBasis(model; Ecut, kgrid = fill(nkpts, 3))
    scfres = self_consistent_field(basis; tol = tol / 10)
    return scfres.energies.total
end

# And since nkpts only takes discrete values of `k` in `[k,k,k]`, we define a custom function to define its convergence for a given desired tolerance `tol`
get_converged_k(energies; tol) = findfirst([abs(i - j) < tol for (i, j) in zip(energies, energies[2:end])])

# Now we can use `get_converged_k()` and `compute_convergence()` to get the converging nkpts
energies_w_diff_k = [compute_convergence(Ecut, k) for k in 1:7] # change to 10 to get full convergence like in plot
k_conv = get_converged_k(energies_w_diff_k; tol) # 5.0

# and plot it:
plot(energies_w_diff_k, dpi = 300, lw = 3, xlabel = "k-grid", ylabel = "Energy/Atom", label = "k")
scatter!(energies_w_diff_k, label = "Data points")

#md # ```@raw html
#md # <img src="../assets/kgrid.png" width=600 height=400 />
#md # ```
#nb # <img src="../assets/kgrid.png" width=600 height=400 />

# this plot may look a little different, but if you change k to 10, you should get the identical plot.

# And then use `optimize()` from `Optim.jl` to do the same for `Ecut`
opt_res = optimize(Ecut -> compute_convergence(Ecut, nkpts, tol), 7, 22)
E_cut_conv = Optim.minimizer(opt_res)
E_cut_conv_H = auconvert(u"eV", E_cut_conv)

# and finally plot Ecut:
Ecut_convergence = [compute_convergence(ecut_i, nkpts) for ecut_i in 7:14] # change to 22 to get full convergence like in plot
plot(Ecut_convergence, dpi = 300, lw = 3, xlabel = "Energy cutoff (Ha)", ylabel = "Energy/Atom", label = "Ecut")
scatter!(Ecut_convergence, label = "Data points")

#md # ```@raw html
#md # <img src="../assets/ecut.png" width=600 height=400 />
#md # ```
#nb # <img src="../assets/ecut.png" width=600 height=400 />

# Once again, a more careful convergence leads to a different plot. 
# By changing `for i in 7:14` to `for i in 7:22`, you should get the same plot.
