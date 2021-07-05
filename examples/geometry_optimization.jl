# # Geometry optimization
#
# We use the DFTK and Optim packages in this example to find the minimal-energy
# bond length of the ``H_2`` molecule. We setup ``H_2`` in an
# LDA model just like in the [Tutorial](@ref) for silicon.
using DFTK
using Optim
using LinearAlgebra
using Printf

kgrid = [1, 1, 1]       # k-Point grid
Ecut = 5                # kinetic energy cutoff in Hartree
tol = 1e-8              # tolerance for the optimization routine
a = 10                  # lattice constant in Bohr
lattice = a * Diagonal(ones(3))
H = ElementPsp(:H, psp=load_psp("hgh/lda/h-q1"));

# We define a blochwave and a density to be used as global variables so that we
# can transfer the solution from one iteration to another and therefore reduce
# the optimization time.

ψ = nothing
ρ = nothing

# First, we create a function that computes the solution associated to the
# position ``x \in \mathbb{R}^6`` of the atoms in reduced coordinates
# (cf. [Reduced and cartesian coordinates](@ref) for more
# details on the coordinates system).
# They are stored as a vector: `x[1:3]` represents the position of the
# first atom and `x[4:6]` the position of the second.
# We also update `ψ` and `ρ` for the next iteration.

function compute_scfres(x)
    atoms = [H => [x[1:3], x[4:6]]]
    model = model_LDA(lattice, atoms)
    basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
    global ψ, ρ
    if ρ === nothing
        ρ = guess_density(basis)
    end
    scfres = self_consistent_field(basis; ψ=ψ, ρ=ρ,
                                   tol=tol / 10, callback=info->nothing)
    ψ = scfres.ψ
    ρ = scfres.ρ
    scfres
end;

# Then, we create the function we want to optimize: `fg!` is used to update the
# value of the objective function `F`, namely the energy, and its gradient `G`,
# here computed with the forces (which are, by definition, the negative gradient
# of the energy).

function fg!(F, G, x)
    scfres = compute_scfres(x)
    if G != nothing
        grad = compute_forces(scfres)
        G .= -[grad[1][1]; grad[1][2]]
    end
    scfres.energies.total
end;

# Now, we can optimize on the 6 parameters `x = [x1, y1, z1, x2, y2, z2]` in
# reduced coordinates, using `LBFGS()`, the default minimization algorithm
# in Optim. We start from `x0`, which is a first guess for the coordinates. By
# default, `optimize` traces the output of the optimization algorithm during the
# iterations. Once we have the minimizer `xmin`, we compute the bond length in
# cartesian coordinates.

x0 = vcat(lattice \ [0., 0., 0.], lattice \ [1.4, 0., 0.])
xres = optimize(Optim.only_fg!(fg!), x0, LBFGS(),
                Optim.Options(show_trace=true, f_tol=tol))
xmin = Optim.minimizer(xres)
dmin = norm(lattice*xmin[1:3] - lattice*xmin[4:6])
@printf "\nOptimal bond length for Ecut=%.2f: %.3f Bohr\n" Ecut dmin

# We used here very rough parameters to generate the example and
# setting `Ecut` to 10 Ha yields a bond length of 1.523 Bohr,
# which [agrees with ABINIT](https://docs.abinit.org/tutorial/base1/).
#
# !!! note "Degrees of freedom"
#     We used here a very general setting where we optimized on the 6 variables
#     representing the position of the 2 atoms and it can be easily extended
#     to molecules with more atoms (such as ``H_2O``). In the particular case
#     of ``H_2``, we could use only the internal degree of freedom which, in
#     this case, is just the bond length.
