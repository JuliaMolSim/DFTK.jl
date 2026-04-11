@testitem "Truncated Coulomb: Model periodicity field" tags=[:minimal] begin
    using DFTK
    using LinearAlgebra

    # Orthogonal cubic cell - fully periodic (default)
    lattice = 10 * I(3)
    atoms = [ElementCoulomb(:H)]
    positions = [[0.5, 0.5, 0.5]]
    model = Model(lattice, atoms, positions; terms=[Kinetic()], n_electrons=1,
                  spin_polarization=:spinless)
    @test model.periodicity == (true, true, true)
    @test is_fully_periodic_electrostatics(model)
    @test n_periodic_electrostatics(model) == 3

    # Fully isolated electrostatics
    model_iso = Model(lattice, atoms, positions;
                      terms=[Kinetic()], n_electrons=1, spin_polarization=:spinless,
                      symmetries=false,
                      periodicity=(:wavefunctions_only, :wavefunctions_only, :wavefunctions_only))
    @test model_iso.periodicity == (:wavefunctions_only, :wavefunctions_only, :wavefunctions_only)
    @test !is_fully_periodic_electrostatics(model_iso)
    @test n_periodic_electrostatics(model_iso) == 0
    @test is_wavefunctions_periodic(model_iso, 1)

    # `false` is accepted (reserved for future fully-isolated wavefunctions)
    model_false = Model(lattice, atoms, positions;
                        terms=[Kinetic()], n_electrons=1, spin_polarization=:spinless,
                        symmetries=false, periodicity=(false, false, false))
    @test model_false.periodicity == (false, false, false)

    # Mixed (2D slab): isolated direction must be orthogonal to the two periodic ones
    lat2D = diagm([10.0, 12.0, 20.0])
    model_2D = Model(lat2D, atoms, positions;
                     terms=[Kinetic()], n_electrons=1, spin_polarization=:spinless,
                     symmetries=false, periodicity=(true, true, :wavefunctions_only))
    @test n_periodic_electrostatics(model_2D) == 2

    # Non-orthogonal lattice should error when mixing periodic and isolated directions
    lat_nort = [10.0 5.0 0.0; 0.0 10.0 0.0; 0.0 0.0 10.0]
    @test_throws ErrorException Model(lat_nort, atoms, positions;
                                      terms=[Kinetic()], n_electrons=1,
                                      spin_polarization=:spinless, symmetries=false,
                                      periodicity=(true, :wavefunctions_only, true))

    # Invalid periodicity value should error
    @test_throws ErrorException Model(lattice, atoms, positions;
                                      terms=[Kinetic()], n_electrons=1,
                                      spin_polarization=:spinless, symmetries=false,
                                      periodicity=(:bad, true, true))
end

@testitem "Truncated Coulomb: kernel values" tags=[:minimal] begin
    using DFTK
    using DFTK: truncated_coulomb_fourier, truncated_coulomb_radius
    using LinearAlgebra

    L = 20.0
    lattice = L * I(3)
    atoms = [ElementCoulomb(:H)]
    positions = [[0.5, 0.5, 0.5]]

    # 3D periodic: 4π/G²
    model_3D = Model(lattice, atoms, positions; terms=[Kinetic()], n_electrons=1,
                     spin_polarization=:spinless, symmetries=false)
    G1 = [2π/L, 0.0, 0.0]
    @test truncated_coulomb_fourier(G1, model_3D) ≈ 4π / sum(abs2, G1)
    @test truncated_coulomb_fourier(zeros(3), model_3D) ≈ 0.0  # neutralising background

    # 0D isolated: 4π/G² * (1 - cos(|G|R)), G=0 → 2πR²
    model_0D = Model(lattice, atoms, positions; terms=[Kinetic()], n_electrons=1,
                     spin_polarization=:spinless, symmetries=false,
                     periodicity=(:wavefunctions_only, :wavefunctions_only, :wavefunctions_only))
    R = truncated_coulomb_radius(model_0D)
    @test R ≈ L / 2  # automatic choice: half minimum box dimension

    Gnorm = 2π / L
    Gc = [Gnorm, 0.0, 0.0]
    vc = truncated_coulomb_fourier(Gc, model_0D)
    @test vc ≈ 4π * (1 - cos(Gnorm * R)) / Gnorm^2

    vc0 = truncated_coulomb_fourier(zeros(3), model_0D)
    @test vc0 ≈ 2π * R^2

    # 2D slab: 4π/G²*(1 - exp(-G_∥ R) cos(G_z R))
    lattice_slab = diagm([10.0, 10.0, 20.0])
    model_slab = Model(lattice_slab, atoms, positions; terms=[Kinetic()], n_electrons=1,
                       spin_polarization=:spinless, symmetries=false,
                       periodicity=(true, true, :wavefunctions_only))
    R_slab = 10.0  # half of 20 Bohr in the isolated z direction
    Gc_slab = [2π/10, 0.0, 2π/20]  # G with both in-plane and out-of-plane components
    Gsq = sum(abs2, Gc_slab)
    Gpar = Gc_slab[1]
    Gz   = Gc_slab[3]
    vc_slab = truncated_coulomb_fourier(Gc_slab, model_slab)
    @test vc_slab ≈ 4π / Gsq * (1 - exp(-Gpar * R_slab) * cos(Gz * R_slab))
end

@testitem "Truncated Coulomb: direct Ewald pair sum" tags=[:minimal] begin
    using DFTK
    using DFTK: energy_forces_ewald
    using LinearAlgebra

    # H₂: 2 protons at distance 1.4 Bohr — simple repulsive ion-ion test
    L = 20.0
    lattice = L * I(3)
    d = 1.4
    positions = [[0.5 - d/2/L, 0.5, 0.5], [0.5 + d/2/L, 0.5, 0.5]]
    charges = [1.0, 1.0]

    (; energy, forces) = energy_forces_ewald(lattice, charges, positions;
                                             periodicity=(false, false, false))
    # Ion-ion Coulomb: Z²/d = 1/1.4
    @test energy ≈ 1.0 / d atol=1e-12

    # Forces in reduced coordinates: convert to Cartesian via inv(L')
    # F_red = L' * F_cart  →  F_cart = L'^{-T} * F_red = F_red / L  for cubic
    F1_cart = forces[1] / L
    @test F1_cart[1] < 0                       # atom 1 pushed left (−x) by atom 2
    @test abs(F1_cart[1]) ≈ 1.0 / d^2 atol=1e-12
end

@testitem "Truncated Coulomb: ionic potential G=0 value" tags=[:minimal] begin
    using DFTK
    using DFTK: compute_local_potential, truncated_coulomb_fourier, truncated_coulomb_radius
    using LinearAlgebra

    L = 20.0
    lattice = L * I(3)
    # Single H atom (Z=1) at cell centre
    atoms = [ElementCoulomb(:H)]
    positions = [[0.5, 0.5, 0.5]]
    model = Model(lattice, atoms, positions;
                  terms=[Kinetic(), AtomicLocal()],
                  n_electrons=1, spin_polarization=:spinless, symmetries=false,
                  periodicity=(:wavefunctions_only, :wavefunctions_only, :wavefunctions_only))
    basis = PlaneWaveBasis(model; Ecut=5.0, kgrid=(1, 1, 1))

    # Compute the truncated ionic potential in real space, then FFT back.
    V_real = compute_local_potential(basis)
    V_fourier = fft(basis, V_real)

    R = truncated_coulomb_radius(model)         # = L/2 = 10 Bohr
    vc0 = truncated_coulomb_fourier(zeros(3), model)  # = 2π R²
    Ω = model.unit_cell_volume
    # At G=0 the ionic potential for a truncated Coulomb is ∫V dr / sqrt(Ω)
    # = -Z * vc0 / sqrt(Ω)  (since V_short(G=0) = 0 for bare Coulomb)
    expected_G0 = -1.0 * vc0 / sqrt(Ω)
    @test real(V_fourier[1]) ≈ expected_G0 atol=1e-8

    # Verify that the periodic (non-truncated) potential has V_ion(G=0) = 0
    model_per = Model(lattice, atoms, positions;
                      terms=[Kinetic(), AtomicLocal()],
                      n_electrons=1, spin_polarization=:spinless, symmetries=false)
    basis_per = PlaneWaveBasis(model_per; Ecut=5.0, kgrid=(1, 1, 1))
    V_per = compute_local_potential(basis_per)
    V_per_fourier = fft(basis_per, V_per)
    @test abs(V_per_fourier[1]) < 1e-12   # zero by compensating-background convention
end

@testitem "Truncated Coulomb: isolated dipole convergence" tags=[:slow] begin
    using DFTK
    using LinearAlgebra

    # Two Gaussian atoms with different potential strengths create an asymmetric
    # external potential and hence a non-zero electronic dipole moment. We verify
    # that the SCF energy and dipole converge rapidly with box size under the
    # truncated Coulomb treatment (no spurious image interactions).
    #
    # For ElementGaussian the ionic charge is zero, so only the Hartree
    # (electron-electron) truncation matters here.

    function compute_energy_dipole(L; Ecut=4.0,
                                   periodicity=(:wavefunctions_only,
                                                :wavefunctions_only,
                                                :wavefunctions_only))
        d = 1.4   # atomic separation (Bohr)
        lattice = L * I(3)
        # Stronger Gaussian (α=1.5) left of centre, weaker (α=0.5) right → electron
        # density concentrated slightly left → dipole relative to centre is negative
        atoms = [ElementGaussian(1.5, 0.5; symbol=:A),
                 ElementGaussian(0.5, 0.5; symbol=:B)]
        positions = [[0.5 - d/(2L), 0.5, 0.5],
                     [0.5 + d/(2L), 0.5, 0.5]]

        # ElementGaussian has charge_ionic=0; disable the charge-neutrality check
        # and set n_electrons manually.
        model = Model(lattice, atoms, positions;
                      terms=[Kinetic(), AtomicLocal(), Hartree()],
                      n_electrons=1, spin_polarization=:spinless,
                      symmetries=false, periodicity,
                      disable_electrostatics_check=true)
        basis = PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1))
        scfres = self_consistent_field(basis; tol=1e-8, callback=identity)

        # Dipole moment along x relative to cell centre: d_x = ∫(x - L/2) ρ(r) dr
        ρtot = Array(total_density(scfres.ρ))
        x_frac = [r[1] for r in r_vectors(basis)]  # fractional x ∈ [0, 1)
        dipole_x = sum((x_frac .- 0.5) .* ρtot) * basis.dvol * L  # in Bohr

        scfres.energies.total, dipole_x
    end

    # Three box sizes: energy and dipole should be stable with truncated Coulomb
    E15, d15 = compute_energy_dipole(15.0)
    E20, d20 = compute_energy_dipole(20.0)
    E25, d25 = compute_energy_dipole(25.0)

    # Energy should be converged: change < 0.5 mHa between successive box sizes
    @test abs(E20 - E15) < 5e-4
    @test abs(E25 - E20) < 5e-4

    # Dipole should be converged: change < 1 mBohr between successive box sizes
    @test abs(d20 - d15) < 1e-3
    @test abs(d25 - d20) < 1e-3

    # Dipole should be non-zero and negative (electron pulled toward the stronger
    # Gaussian which sits at x < L/2)
    @test d20 < -0.01
end
