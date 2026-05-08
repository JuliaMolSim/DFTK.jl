@testitem "SymOp θ field: composition, inverse, group closure" tags=[:minimal] begin
    using DFTK
    using DFTK: SymOp, Mat3, Vec3, check_group
    using LinearAlgebra

    s_id = SymOp(Mat3{Int}(I), Vec3(0.0, 0.0, 0.0))
    s_tr = SymOp(Mat3{Int}(I), Vec3(0.0, 0.0, 0.0); θ=-1)

    @test s_id.θ == 1
    @test s_tr.θ == -1
    @test isone(s_id)
    @test !isone(s_tr)
    @test s_id != s_tr

    @test (s_id * s_tr).θ == -1
    @test (s_tr * s_id).θ == -1
    @test (s_tr * s_tr).θ == 1

    @test inv(s_id).θ == 1
    @test inv(s_tr).θ == -1

    inv_op = SymOp(-Mat3{Int}(I), Vec3(0.0, 0.0, 0.0))
    inv_tr = SymOp(-Mat3{Int}(I), Vec3(0.0, 0.0, 0.0); θ=-1)
    check_group([s_id, inv_op, s_tr, inv_tr])
end


# Coverage matrix: each (system, basis configuration) pair runs the SCF twice
# (with and without BZ symmetry) and asserts the same battery of checks.
#
# Systems target distinct paths through the symmetry pipeline:
#   - Si non-magnetic: :none with inversion (TRS adds nothing new in BZ orbits)
#   - GaAs equilibrium: :none, no inversion (TRS halves the BZ — headline win)
#   - Rattled GaAs: :none, only identity spatial (pure TRS, exercises TRIM points)
#   - AFM Si  [+1, -1]: :collinear with paired moments → spglib spin_flips=-1
#                        → real θ=-1 partners with W not in θ=+1 set
#   - FM  Si  [+1, +1]: :collinear without antiunitary symops (sanity)
#
# Si is also exercised on a battery of kgrid/kshift combinations to keep the
# kgrid-edge-case coverage that used to live in a separate testitem.
#
# Per-case checks (T1..T4, T6 from the design matrix):
#   T1 SCF equivalence (sym vs no-sym): |ΔE|, |Δρ|, |Δforces| at SCF tolerance
#   T2 unfold_bz round-trip preserves total energy and density
#   T3 sum(basis.kweights) ≈ n_spin_components
#   T4 check_group on basis.symmetries (augmented group closes)
#   T6 length(basis.kpoints) drops where the geometry allows it
@testitem "TRS coverage matrix: SCF equivalence, unfold, group closure" #=
    =#    tags=[:slow] setup=[TestCases] begin
    using DFTK
    using LinearAlgebra

    Ga = ElementPsp(:Ga, load_psp("hgh/lda/Ga-q3"))
    As = ElementPsp(:As, load_psp("hgh/lda/As-q5"))
    a_GaAs       = 10.68
    lattice_GaAs = a_GaAs / 2 * [0 1 1; 1 0 1; 1 1 0]
    pos_GaAs     = [[0, 0, 0], [1/4, 1/4, 1/4]]
    δ            = 0.04
    pos_GaAs_rattle = [[0, 0, 0]      .+ δ * [ 0.3, -0.2,  0.5],
                       [1/4, 1/4, 1/4] .+ δ * [-0.4,  0.1, -0.3]]

    silicon = TestCases.silicon

    silicon_basis_args = [(; kgrid=[2, 2, 2], kshift=[1/2, 0,   0  ]),
                          (; kgrid=[2, 2, 2], kshift=[1/2, 1/2, 0  ]),
                          (; kgrid=[2, 2, 2], kshift=[0,   0,   0  ]),
                          (; kgrid=[3, 2, 3], kshift=[0,   0,   0  ]),
                          (; kgrid=[3, 2, 3], kshift=[0,   1/2, 1/2])]

    systems = [
        (; name = "Si (non-magnetic)",
           lattice = silicon.lattice, atoms = silicon.atoms,
           positions = silicon.positions, magnetic_moments = [],
           Ecut = 5, basis_args = silicon_basis_args,
           expect_halving = false),
        (; name = "GaAs equilibrium (Td, no inversion)",
           lattice = lattice_GaAs, atoms = [Ga, As], positions = pos_GaAs,
           magnetic_moments = [], Ecut = 5,
           basis_args = [(; kgrid=[3, 3, 3])],
           expect_halving = true),
        (; name = "Rattled GaAs (TRS-only)",
           lattice = lattice_GaAs, atoms = [Ga, As], positions = pos_GaAs_rattle,
           magnetic_moments = [], Ecut = 5,
           basis_args = [(; kgrid=[3, 3, 3])],
           expect_halving = true),
        (; name = "AFM Si (paired moments)",
           lattice = silicon.lattice, atoms = silicon.atoms,
           positions = silicon.positions, magnetic_moments = [1, -1],
           Ecut = 5, basis_args = [(; kgrid=[2, 2, 2])],
           expect_halving = false),
        (; name = "FM Si (parallel moments)",
           lattice = silicon.lattice, atoms = silicon.atoms,
           positions = silicon.positions, magnetic_moments = [1, 1],
           Ecut = 5, basis_args = [(; kgrid=[2, 2, 2])],
           expect_halving = false),
    ]

    function run(sys, basis_args; use_symmetries)
        model = model_DFT(sys.lattice, sys.atoms, sys.positions;
                          functionals = LDA(),
                          magnetic_moments = sys.magnetic_moments,
                          symmetries = use_symmetries)
        basis = PlaneWaveBasis(model; Ecut = sys.Ecut, basis_args...)
        ρ0 = isempty(sys.magnetic_moments) ? guess_density(basis) :
                                             guess_density(basis, sys.magnetic_moments)
        scfres = self_consistent_field(basis; tol = 1e-10, ρ = ρ0, callback = identity)
        (; basis, scfres, forces = compute_forces_cart(scfres))
    end

    for sys in systems, basis_args in sys.basis_args
        @testset "$(sys.name) $(NamedTuple(basis_args))" begin
            sym   = run(sys, basis_args; use_symmetries = true)
            nosym = run(sys, basis_args; use_symmetries = false)

            # T4 — augmented symmetry group closes
            DFTK.check_group(sym.basis.symmetries)

            # T3 — kweights normalised to n_spin_components (1 non-collinear, 2 collinear)
            @test sum(sym.basis.kweights) ≈ sym.basis.model.n_spin_components

            # T1 — SCF equivalence vs full-BZ reference
            dvol = sym.basis.dvol
            @test abs(sym.scfres.energies.total - nosym.scfres.energies.total) < 1e-8
            @test norm(sym.scfres.ρ .- nosym.scfres.ρ) * sqrt(dvol)            < 1e-7
            @test maximum(norm.(sym.forces .- nosym.forces))                   < 1e-5

            # T6 — k-count halving smell test where TRS adds new BZ-orbit elements
            if sys.expect_halving
                @test length(sym.basis.kpoints) < length(nosym.basis.kpoints)
            end

            # T2 — unfold_bz round-trip exercises apply_symop on θ=-1 partners and
            # unfold_mapping (with the collinear ↑↔↓ spin swap)
            unfolded = DFTK.unfold_bz(sym.scfres)
            @test abs(unfolded.energies.total - sym.scfres.energies.total)     < 1e-9
            @test norm(unfolded.ρ .- sym.scfres.ρ) * sqrt(dvol)                < 1e-9
        end
    end
end


# Hubbard with paired AFM moments — exercises symmetrize_hubbard_n on θ=-1
# (conj(n_src) + ↑↔↓ source-spin swap) which the matrix above does not touch.
@testitem "TRS / Hubbard with collinear AFM (paired moments)" tags=[:slow] begin
    using DFTK
    using PseudoPotentialData
    using Unitful
    using UnitfulAtomic
    using LinearAlgebra

    a = 7.9
    lattice = a * [1.0  0.5  0.5; 0.5  1.0  0.5; 0.5  0.5  1.0]
    pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
    Ni = ElementPsp(:Ni, pseudopotentials; rcut=10.0)
    O  = ElementPsp(:O,  pseudopotentials; rcut=10.0)
    atoms = [Ni, O, Ni, O]
    positions = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25],
                 [0.5, 0.5, 0.5], [0.75, 0.75, 0.75]]
    magnetic_moments = [2, 0, -2, 0]

    hubbard = Hubbard(OrbitalManifold(Ni, "3D") => 10u"eV")
    model = model_DFT(lattice, atoms, positions; extra_terms=[hubbard],
                      temperature=0.01, functionals=PBE(),
                      smearing=DFTK.Smearing.Gaussian(), magnetic_moments)
    basis = PlaneWaveBasis(model; Ecut=10, kgrid=[2, 2, 2])

    @test count(s -> s.θ == -1, basis.model.symmetries) > 0

    ρ0 = guess_density(basis, magnetic_moments)
    scfres = self_consistent_field(basis; tol=1e-7, ρ=ρ0, callback=identity)

    if mpi_nprocs(basis.comm_kpts) == 1
        scfres_nosym = DFTK.unfold_bz(scfres)
        term_idx = findfirst(t -> isa(t, DFTK.TermHubbard), scfres_nosym.basis.terms)
        nhub_nosym = DFTK.compute_hubbard_n(scfres_nosym.basis.terms[term_idx],
                                             scfres_nosym.basis,
                                             scfres_nosym.ψ, scfres_nosym.occupation)
        @test scfres.hubbard_n ≈ nhub_nosym  rtol=1e-6
    end
end
