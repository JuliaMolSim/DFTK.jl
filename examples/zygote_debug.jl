using DFTK
using Zygote
using FiniteDiff
setup_threading(n_blas=1)

# Specify structure for silicon lattice
a = 10.26
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms     = [Si,        Si        ]
positions = [ones(3)/8, -ones(3)/8]# + 0.1rand(3)]

terms = [
    Kinetic(),
    AtomicLocal(),
    AtomicNonlocal(),
    Ewald(),
    PspCorrection(),
    Hartree(),
]
model = Model(lattice, atoms, positions; terms, symmetries=false)
Ecut = 15
basis = PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1), kshift=(0, 0, 0))

function energy_from_basis(basis)
    is_converged = DFTK.ScfConvergenceDensity(1e-8)
    scfres = self_consistent_field(basis; is_converged)
    # scfres.energies.total
    sum(values(scfres.energies))
end
energy_from_basis(basis)
Zygote.gradient(energy_from_basis, basis)

function forces_from_basis(basis)
    is_converged = DFTK.ScfConvergenceDensity(1e-8)
    scfres = self_consistent_field(basis; is_converged)
    compute_forces(scfres)[1][1]
end
forces_from_basis(basis)
Zygote.gradient(forces_from_basis, basis)

function eigenvalues_from_basis(basis)
    is_converged = DFTK.ScfConvergenceDensity(1e-8)
    scfres = self_consistent_field(basis; is_converged)
    sum(sum, scfres.eigenvalues)
end
Zygote.gradient(eigenvalues_from_basis, basis)

# Comparison to FiniteDiff

function basis_from_lattice(lattice)
    terms = [
        Kinetic(),
        AtomicLocal(),
        AtomicNonlocal(),
        Ewald(),
        PspCorrection(),
        Hartree(),
    ]
    model = Model(lattice, atoms, positions; terms, symmetries=false)
    PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1), kshift=(0, 0, 0))
end

energy_from_lattice(lattice) = energy_from_basis(basis_from_lattice(lattice))
e = energy_from_lattice(lattice)
e_zg, (grad_e_zg,) = Zygote.withgradient(energy_from_lattice, lattice)
grad_e_fd = FiniteDiff.finite_difference_gradient(energy_from_lattice, lattice)
println("energy error ", abs(e - e_zg))
println("energy w.r.t. lattice error ", abs.(grad_e_fd - grad_e_zg))

forces_from_lattice(lattice) = forces_from_basis(basis_from_lattice(lattice)) 
f = forces_from_lattice(lattice)
f_zg, (grad_f_zg,) = Zygote.withgradient(forces_from_lattice, lattice) # TODO debug values
grad_f_fd = FiniteDiff.finite_difference_gradient(forces_from_lattice, lattice)
println("force error ", abs(f - f_zg))
println("force w.r.t. lattice error ", abs.(grad_f_fd - grad_f_zg))

eigenvalues_from_lattice(lattice) = eigenvalues_from_basis(basis_from_lattice(lattice))
ev = eigenvalues_from_lattice(lattice)
ev_zg, (grad_ev_zg,) = Zygote.withgradient(eigenvalues_from_lattice, lattice) # TODO debug values
grad_ev_fd = FiniteDiff.finite_difference_gradient(eigenvalues_from_lattice, lattice)
println("eigenvalues error ", abs(ev - ev_zg))
println("eigenvalues w.r.t. lattice error ", abs.(grad_ev_fd - grad_ev_zg))
