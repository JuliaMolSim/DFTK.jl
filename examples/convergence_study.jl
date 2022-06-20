# # Performing a convergence study
#
# This example shows how to perform a convergence study to find the 
# appropriate kgrid and Ecut for a desired convergence and error tolerance.
#
# Generally, to wisely vary kgrid and Ecut, one can perform a convergence study by
# starting with a reasonable grid or Ecut, picking a property such as energy, and increasing the grid
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

kgrid = [1, 1, 1]       # k-point grid
tol = 1e-2              # tolerance for the optimization routine
Ecut = 490u"eV"
temperature = 0.01
Pt = ElementPsp(:Pt, psp=load_psp("hgh/lda/Pt-q10"));
atoms = [Pt]
position = [zeros(3)]
a = 5.00 # in Bohr
lattice = a * I(3)

# Then we define a function to compute the total energy of the system for varying values of `Ecut` and `kgrid`
function compute_Ecut_convergence(Ecut)
    model = model_LDA(lattice, atoms,position; temperature)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    scfres = self_consistent_field(basis; tol=tol / 10)
    return scfres.energies.total
end

function compute_kgrid_convergence(k; lat=:sc)
    model = model_LDA(lattice, atoms,position; temperature)
    basis = PlaneWaveBasis(model; Ecut, kgrid=[Int64(ceil(k)) for _ in 1:3])
    scfres = self_consistent_field(basis; tol=tol / 10)
    return scfres.energies.total
end

# And since kgrid only takes discrete values of `k` in `[k,k,k]`, we define a custom function to define its convergence for a given desired tolerance `tol`
function get_converged_k(energies; tol)
    for i in eachindex(energies)
        if abs(energies[i] - energies[i+1]) < tol
            return i
        end
    end
end

# Now we can use `get_converged_k()` and `compute_kgrid_convergence()` to get the converging kgrid
energies_w_diff_k = [compute_kgrid_convergence(k) for k in 1:10]
k_conv = get_converged_k(energies_w_diff_k; tol) # 5.0

# and plot it:
plot(energies_w_diff_k, dpi=300, lw=3, xlabel="k-grid", ylabel="Energy/Atom", label="k")
scatter!(energies_w_diff_k,label="Data points")

#md # ```@raw html
#md # <img src="../assets/kgrid.png" width=600 height=400 />
#md # ```

# And then use `optimize()` from `Optim.jl` to do the same for `Ecut`
opt_res = optimize(compute_Ecut_convergence, 7, 22) 
E_cut_conv = Optim.minimizer(opt_res)
E_cut_conv_H = auconvert(u"eV", E_cut_conv)

# and finally plot Ecut:
Ecut_convergence = [compute_Ecut_convergence(i) for i in  7:22]
plot(Ecut_convergence, dpi=300, lw=3, xlabel = "Energy cutoff (Ha)", ylabel="Energy/Atom", label="Ecut")
scatter!(Ecut_convergence,label="Data points")

#md # ```@raw html
#md # <img src="../assets/ecut.png" width=600 height=400 />
#md # ```