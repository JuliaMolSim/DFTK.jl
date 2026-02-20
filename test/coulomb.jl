@testitem "Reference tests for exx implementations" tags=[:exx] setup=[TestCases] begin
    using DFTK
    using DFTK: exx_energy_only
    using .TestCases: silicon

    # TODO: This is a bad test, better test properties, such as the fact that
    #       the Coulomb energy converges to the same value as the supercell / k-grid
    #       converges to larger values.

    Si = ElementPsp(14, load_psp(silicon.psp_upf))
    model  = model_DFT(silicon.lattice, [Si, Si], silicon.positions; functionals=PBE())
    basis  = PlaneWaveBasis(model; Ecut=10, kgrid=(1, 1, 1))
    scfres = self_consistent_field(basis; tol=1e-10, #=callback=identity=#)

    n_occ = 4
    kpt  = basis.kpoints[1]
    ψk   = scfres.ψ[1][:, 1:n_occ]
    occk = scfres.occupation[1][1:n_occ]
    ψk_real = similar(ψk, complex(T), basis.fft_size..., n_occ)
    @views for n = 1:n_occ
        ifft!(ψk_real[:, :, :, n], basis, kpt, ψk[:, n])
    end

    k_neglect = compute_coulomb_kernel(basis; coulomb_kernel_model=NeglectSingularity())
    @testset "NeglectSingularity" begin
        E_neglect = exx_energy_only(basis, kpt, k_neglect, ψk_real, occk)
        E_ref = 0.0
        @test abs(E_neglect - E_ref) < 1e-6
    end



end





# TODO: Put some tests here comparing the different coulomb models
#       e.g. all coulomb models should have the same large-q behaviour
#
# Test type stability

# Test reference values for all kernels ?

#=
Test that probe charge model gives the same elements but the first

Test that spherically truncated gives a truncated function ?
=#
