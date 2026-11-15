@testitem "Hamiltonian derivative wrt. k" tags=[:minimal] setup=[TestCases] begin
    using DFTK
    using LinearAlgebra

    function apply_kshift(kshift, basis, ψ; ρ, τ)
        kcoords = map(basis.kcoords_global) do kcoord
            kcoord + kshift
        end
        kpoints_shifted = ExplicitKpoints(kcoords, basis.kweights_global)
        basis_shifted = PlaneWaveBasis(basis.model; Ecut=basis.Ecut, kgrid=kpoints_shifted)
        ham = Hamiltonian(basis_shifted; ρ, τ)
        ham * ψ
    end

    for spin in [:none, :collinear]
        @testset "Spin polarization: $spin" begin
            testcase = TestCases.silicon
            Si = ElementPsp(testcase.atnum, load_psp(testcase.psp_upf))
            magnetic_moments = spin == :collinear ? [0.5, -0.5] : []
            model = model_DFT(testcase.lattice, [Si, Si], testcase.positions;
                              functionals=r2SCAN(), magnetic_moments, symmetries=false)
            basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2])
            scfres = self_consistent_field(basis;
                                           ρ=guess_density(basis, magnetic_moments),
                                           maxiter=3, callback=identity)

            function test_term(term)
                @testset "Term: $term" begin
                    model = Model(testcase.lattice, [Si, Si], testcase.positions; terms=[term],
                                  magnetic_moments, symmetries=false)
                    basis = PlaneWaveBasis(model; basis.Ecut, basis.kgrid)
                    @assert length(basis.terms) == 1

                    δk = randn(3)
                    δk ./= norm(δk)
                    δHδk_ad = DFTK.compute_Hk_derivative(basis, scfres.ψ, δk; scfres.ρ, scfres.τ)

                    h = 1e-4
                    δHδk_fd = (apply_kshift( h * δk, basis, scfres.ψ; scfres.ρ, scfres.τ)
                             - apply_kshift(-h * δk, basis, scfres.ψ; scfres.ρ, scfres.τ)) / 2h

                    for (δ_ad, δ_fd, kpt) in zip(δHδk_ad, δHδk_fd, basis.kpoints)
                        @test δ_ad ≈ δ_fd rtol=1e-8
                    end
                end
            end

            test_term(Kinetic())
            test_term(AtomicLocal())
            test_term(AtomicNonlocal())
            test_term(PBE())
            test_term(r2SCAN())
        end
    end
end