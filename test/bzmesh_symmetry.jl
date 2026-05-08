@testitem "Symmetrization and not symmetrization yields the same density and energy" #=
    =#    setup=[TestCases] begin
    using DFTK
    using LinearAlgebra
    testcase = TestCases.silicon

    args = ((; kgrid=[2, 2, 2], kshift=[1/2, 0, 0]),
            (; kgrid=[2, 2, 2], kshift=[1/2, 1/2, 0]),
            (; kgrid=[2, 2, 2], kshift=[0, 0, 0]),
            (; kgrid=[3, 2, 3], kshift=[0, 0, 0]),
            (; kgrid=[3, 2, 3], kshift=[0, 1/2, 1/2]))
    for case in args
        model_nosym = model_DFT(testcase.lattice, testcase.atoms, testcase.positions;
                                functionals=LDA(), symmetries=false)
        basis = PlaneWaveBasis(model_nosym; Ecut=5, case...)
        DFTK.check_group(basis.symmetries)

        scfres = self_consistent_field(basis; is_converged=DFTK.ScfConvergenceDensity(1e-10))
        ρ1 = scfres.ρ
        E1 = scfres.energies.total

        model_sym = model_DFT(testcase.lattice, testcase.atoms, testcase.positions;
                              functionals=LDA())
        basis = PlaneWaveBasis(model_sym; Ecut=5, case...)
        DFTK.check_group(basis.symmetries)
        scfres = self_consistent_field(basis; is_converged=DFTK.ScfConvergenceDensity(1e-10))
        ρ2 = scfres.ρ
        E2 = scfres.energies.total

        @test abs(E1 - E2) < 1e-10
        @test norm(ρ1 - ρ2) .* sqrt(basis.dvol) < 1e-8
    end
end


@testitem "SymOp θ field: composition, inverse, group closure" tags=[:minimal] begin
    using DFTK
    using DFTK: SymOp, Mat3, Vec3, check_group
    using LinearAlgebra

    s_id = SymOp(Mat3{Int}(I), Vec3(0.0, 0.0, 0.0))
    s_tr = SymOp(Mat3{Int}(I), Vec3(0.0, 0.0, 0.0); θ=-1)

    @test s_id.θ == 1
    @test s_tr.θ == -1
    @test isone(s_id)
    @test !isone(s_tr)         # antiunitary identity is not the group identity
    @test s_id != s_tr

    # θ multiplies under composition
    @test (s_id * s_tr).θ == -1
    @test (s_tr * s_id).θ == -1
    @test (s_tr * s_tr).θ == 1

    # inverse preserves θ (antiunitary inverse is antiunitary)
    @test inv(s_id).θ == 1
    @test inv(s_tr).θ == -1

    # Augmented group closes
    inv_op = SymOp(-Mat3{Int}(I), Vec3(0.0, 0.0, 0.0))                 # spatial inversion
    inv_tr = SymOp(-Mat3{Int}(I), Vec3(0.0, 0.0, 0.0); θ=-1)
    check_group([s_id, inv_op, s_tr, inv_tr])
end


# Helper: GaAs (zinc-blende) — no inversion, so TRS halves the irreducible BZ.
@testitem "TRS k-point reduction: GaAs equilibrium" tags=[:slow] begin
    using DFTK
    using LinearAlgebra

    Ga = ElementPsp(:Ga, load_psp("hgh/lda/Ga-q3"))
    As = ElementPsp(:As, load_psp("hgh/lda/As-q5"))
    a = 10.68  # Bohr
    lattice  = a / 2 * [0 1 1; 1 0 1; 1 1 0]
    atoms    = [Ga, As]
    positions = [[0, 0, 0], [1/4, 1/4, 1/4]]

    function run(use_symmetries)
        model = model_DFT(lattice, atoms, positions;
                          functionals=LDA(), symmetries=use_symmetries)
        basis = PlaneWaveBasis(model; Ecut=10, kgrid=[4, 4, 4])
        scfres = self_consistent_field(basis; tol=1e-10, callback=identity)
        forces = compute_forces_cart(scfres)
        (; basis, scfres, forces)
    end

    nosym = run(false)
    sym   = run(true)

    n_sp = count(s -> s.θ == +1, sym.basis.model.symmetries)
    n_tr = count(s -> s.θ == -1, sym.basis.model.symmetries)
    @test n_sp == 24                      # Td (24 unitary symops including identity)
    @test n_tr == 24                      # full TRS partner set
    @test sum(sym.basis.kweights) ≈ 1
    @test length(sym.basis.kpoints) < length(nosym.basis.kpoints)

    @test abs(nosym.scfres.energies.total - sym.scfres.energies.total) < 1e-8
    @test maximum(abs.(nosym.scfres.ρ .- sym.scfres.ρ)) < 1e-7
    @test maximum(norm.(nosym.forces .- sym.forces)) < 1e-6
end


@testitem "TRS k-point reduction: rattled GaAs (TRS-only)" tags=[:slow] begin
    using DFTK
    using LinearAlgebra

    Ga = ElementPsp(:Ga, load_psp("hgh/lda/Ga-q3"))
    As = ElementPsp(:As, load_psp("hgh/lda/As-q5"))
    a = 10.68
    lattice  = a / 2 * [0 1 1; 1 0 1; 1 1 0]
    atoms    = [Ga, As]
    δ = 0.04
    positions = [[0, 0, 0]      .+ δ * [ 0.3, -0.2,  0.5],
                 [1/4, 1/4, 1/4] .+ δ * [-0.4,  0.1, -0.3]]

    function run(use_symmetries)
        model = model_DFT(lattice, atoms, positions;
                          functionals=LDA(), symmetries=use_symmetries)
        basis = PlaneWaveBasis(model; Ecut=10, kgrid=[4, 4, 4])
        scfres = self_consistent_field(basis; tol=1e-10, callback=identity)
        forces = compute_forces_cart(scfres)
        (; basis, scfres, forces)
    end

    nosym = run(false)
    sym   = run(true)

    n_sp = count(s -> s.θ == +1, sym.basis.model.symmetries)
    n_tr = count(s -> s.θ == -1, sym.basis.model.symmetries)
    @test n_sp == 1               # only identity remains spatially
    @test n_tr == 1               # one TRS partner

    # TRIM points (2k ≡ 0 mod lattice) are TRS-invariant and not halved. For [4,4,4]
    # there are 8 such points so the count is (4³ + 8)/2 = 36, not 32.
    @test length(sym.basis.kpoints) == 36
    @test sum(sym.basis.kweights) ≈ 1

    # Genuinely non-zero forces from broken inversion
    @test maximum(norm.(nosym.forces)) > 1e-3

    @test abs(nosym.scfres.energies.total - sym.scfres.energies.total) < 1e-8
    @test maximum(abs.(nosym.scfres.ρ .- sym.scfres.ρ)) < 1e-7
    @test maximum(norm.(nosym.forces .- sym.forces)) < 1e-6
end


@testitem "TRS k-point reduction: unfold_bz round-trip on GaAs" tags=[:slow] begin
    using DFTK
    using LinearAlgebra

    Ga = ElementPsp(:Ga, load_psp("hgh/lda/Ga-q3"))
    As = ElementPsp(:As, load_psp("hgh/lda/As-q5"))
    a = 10.68
    lattice  = a / 2 * [0 1 1; 1 0 1; 1 1 0]
    atoms    = [Ga, As]
    positions = [[0, 0, 0], [1/4, 1/4, 1/4]]

    model = model_DFT(lattice, atoms, positions; functionals=LDA())
    basis = PlaneWaveBasis(model; Ecut=10, kgrid=[4, 4, 4])
    scfres = self_consistent_field(basis; tol=1e-10, callback=identity)

    # unfold_bz exercises unfold_mapping (now θ-aware) and apply_symop on θ=-1 partners.
    # Its internal @assert checks that the energy on the unfolded basis matches.
    unfolded = DFTK.unfold_bz(scfres)
    @test length(unfolded.basis.kpoints) > length(scfres.basis.kpoints)
    @test abs(unfolded.energies.total - scfres.energies.total) < 1e-9
    @test maximum(abs.(unfolded.ρ .- scfres.ρ)) < 1e-10
end


@testitem "TRS / collinear AFM: spglib spin_flips path" #=
    =#    setup=[TestCases] begin
    using DFTK
    using DFTK: SymOp
    using LinearAlgebra

    silicon = TestCases.silicon

    # Antiparallel moments on the two equivalent Si sublattices. Symmetries that map
    # the two sublattices into each other must come with a spin flip (θ=-1).
    magnetic_moments = [1, -1]
    model = model_DFT(silicon.lattice, silicon.atoms, silicon.positions;
                      functionals=LDA(), magnetic_moments)

    n_unitary    = count(s -> s.θ == +1, model.symmetries)
    n_antiunit   = count(s -> s.θ == -1, model.symmetries)
    @test n_antiunit > 0                        # spglib must return some spin_flips==-1
    @test n_unitary  > 0
    @test sum(s -> s.θ == 0, model.symmetries) == 0

    # Build a basis and verify k-weights and the symop group closes.
    basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2])
    DFTK.check_group(basis.symmetries)
    @test sum(basis.kweights) ≈ basis.model.n_spin_components
end
