@testitem "derivative wrt. k" tags=[:minimal] setup=[TestCases] begin
    using DFTK
    using LinearAlgebra

    function apply_kshift(kshift, basis, ρ, ψ)
        # TODO: likely not mpi compatible yet
        kcoords = map(basis.kpoints) do kpt
            kpt.coordinate + kshift
        end
        kpoints_shifted = ExplicitKpoints(kcoords, basis.kweights)
        basis_shifted = PlaneWaveBasis(basis.model; Ecut=basis.Ecut, kgrid=kpoints_shifted)
        ham = Hamiltonian(basis_shifted; ρ)
        ham * ψ
    end

    for spin in [:none] # TODO: , :collinear]
        @testset "Spin polarization: $spin" begin
            # Build a reasonable density from a silicon model
            testcase = TestCases.silicon
            Si = ElementPsp(testcase.atnum, load_psp(testcase.psp_upf))
            magnetic_moments = spin == :collinear ? [0.5, -0.5] : []
            model = model_DFT(testcase.lattice, [Si, Si], testcase.positions;
                              functionals=PBE(), magnetic_moments, symmetries=false)
            basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2])
            scfres = self_consistent_field(basis;
                                           ρ=guess_density(basis, magnetic_moments),
                                           callback=identity)
            
            function test_term(term)
                @testset "Term: $term" begin
                    model = Model(testcase.lattice, [Si, Si], testcase.positions; terms=[term],
                                magnetic_moments, symmetries=false)
                    basis = PlaneWaveBasis(model; basis.Ecut, basis.kgrid)
                    @assert length(basis.terms) == 1
                    
                    δHδk_ad = DFTK.compute_δHδk(basis, scfres.ψ; scfres.ρ)

                    h = 1e-4
                    for α in 1:3
                        dk = zeros(eltype(basis), 3)
                        dk[α] = 1
                        δHδk_fd = (apply_kshift( h * dk, basis, scfres.ρ, scfres.ψ)
                                 - apply_kshift(-h * dk, basis, scfres.ρ, scfres.ψ)) / 2h
                        
                        for (δ_ad, δ_fd, kpt) in zip(δHδk_ad[α], δHδk_fd, basis.kpoints)
                            @test δ_ad ≈ δ_fd rtol=1e-8
                            if !isapprox(δ_ad, δ_fd, rtol=1e-8)
                                relerror = norm(δ_ad - δ_fd) / norm(δ_fd)
                                @show kpt.coordinate relerror
                            end
                        end
                    end
                end
            end

            test_term(Kinetic())
            test_term(AtomicLocal())
            test_term(AtomicNonlocal())
        end
    end
end