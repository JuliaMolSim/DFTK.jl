using DFTK
using Zygote
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

# Test primal
let terms = [Kinetic(), AtomicLocal(), Ewald(), PspCorrection(), Hartree(),]
    model = Model(lattice, atoms, positions; terms, symmetries=false, temperature=1e-3)
    basis = PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1), kshift=(0, 0, 0))
    energy = self_consistent_field(basis; tol=1e-14).energies.total
    reference = -19.629878507271652  # From master
    diff = abs(energy - reference)
    if diff > 2e-8
        @error reference energy diff
    end
end
let terms = terms
    model = Model(lattice, atoms, positions; terms, symmetries=false, temperature=1e-3)
    basis = PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1), kshift=(0, 0, 0))
    energy = self_consistent_field(basis; tol=1e-14).energies.total
    reference = -4.821586293957623  # From master
    diff = abs(energy - reference)
    if diff > 2e-8
        @error reference energy diff
    end
end

function energy_from_basis(basis)
    is_converged = DFTK.ScfConvergenceDensity(1e-8)
    scfres = self_consistent_field(basis; is_converged)
    scfres.energies.total
end
energy_from_basis(basis)
Zygote.gradient(energy_from_basis, basis) # TODO

function forces_from_basis(basis)
    is_converged = DFTK.ScfConvergenceDensity(1e-8)
    scfres = self_consistent_field(basis; is_converged)
    compute_forces(scfres)[1][1]
end
forces_from_basis(basis)
Zygote.gradient(forces_from_basis, basis) # TODO