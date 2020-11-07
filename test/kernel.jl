using DFTK
using Test

include("testcases.jl")

function test_kernel_unpolarized(termtype)
    Ecut=2
    kgrid = [2, 2, 2]
    testcase = silicon
    ε = 1e-8

    @testset "Kernel $(typeof(termtype))" begin
        spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
        model = Model(testcase.lattice; atoms=[spec => testcase.positions],
                      terms=[termtype])
        basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
        term  = only(basis.terms)

        ρ0 = guess_density(basis)
        dρ = from_real(basis, randn(size(ρ0)))

        ρ_minus = ρ0 - ε * dρ
        ρ_plus  = ρ0 + ε * dρ
        pot_minus = DFTK.ene_ops(term, nothing, nothing; ρ=ρ_minus).ops[1].potential
        pot_plus  = DFTK.ene_ops(term, nothing, nothing; ρ=ρ_plus).ops[1].potential
        dV = (pot_plus - pot_minus) / (2ε)

        dV_apply = DFTK.apply_kernel(term, dρ; ρ=ρ0)[1]
        kernel = DFTK.compute_kernel(term; ρ=ρ0)
        dV_compute = reshape(kernel * vec(dρ.real), size(dρ))
        @test norm(dV - dV_apply.real) < 100ε
        @test norm(dV - dV_compute)    < 100ε
    end
end


function test_kernel_collinear(termtype)
    Ecut=2
    kgrid = [2, 2, 2]
    testcase = silicon
    ε = 1e-8

    @testset "Kernel $(typeof(termtype)) (collinear)" begin
        spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
        magnetic_moments = [spec => 2rand(2)]
        model = Model(testcase.lattice; atoms=[spec => testcase.positions],
                      terms=[termtype], magnetic_moments=magnetic_moments,
                      spin_polarization=:collinear)
        basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
        iup   = DFTK.krange_spin(basis, 1)[1]  # First spin-up k-point
        idown = DFTK.krange_spin(basis, 2)[1]  # First spin-down k-point
        term  = only(basis.terms)

        ρ0      = guess_density(basis)
        dρ      = from_real(basis, randn(size(ρ0)))
        ρspin0  = guess_spin_density(basis, magnetic_moments)
        dρspin  = from_real(basis, randn(size(ρspin0)))

        ρ_minus     = ρ0     - ε * dρ
        ρspin_minus = ρspin0 - ε * dρspin
        ρ_plus      = ρ0     + ε * dρ
        ρspin_plus  = ρspin0 + ε * dρspin

        ops_minus = DFTK.ene_ops(term, nothing, nothing; ρ=ρ_minus, ρspin=ρspin_minus).ops
        ops_plus  = DFTK.ene_ops(term, nothing, nothing; ρ=ρ_plus,  ρspin=ρspin_plus).ops
        dVα = (ops_plus[  iup].potential - ops_minus[  iup].potential) / (2ε)
        dVβ = (ops_plus[idown].potential - ops_minus[idown].potential) / (2ε)

        dVα_apply, dVβ_apply = DFTK.apply_kernel(term, dρ, dρspin; ρ=ρ0, ρspin=ρspin0)

        kernel = DFTK.compute_kernel(term; ρ=ρ0, ρspin=ρspin0)
        dρcat  = vcat(vec(dρ.real), vec(dρspin.real))
        dV_matrix = reshape(kernel * dρcat, size(dρ)..., 2)

        @test norm(dVα - dVα_apply.real) < 100ε
        @test norm(dVβ - dVβ_apply.real) < 100ε
        @test norm(dVα - dV_matrix[:, :, :, 1]) < 100ε
        @test norm(dVβ - dV_matrix[:, :, :, 2]) < 100ε
    end
end

test_kernel_unpolarized(PowerNonlinearity(1.2, 2.0))
test_kernel_unpolarized(Hartree())
test_kernel_unpolarized(Xc(:lda_xc_teter93))

test_kernel_collinear(Hartree())
test_kernel_collinear(PowerNonlinearity(1.2, 2.5))
test_kernel_collinear(Xc(:lda_xc_teter93))
