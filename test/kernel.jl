using DFTK
using Test

include("testcases.jl")

function test_kernel_unpolarized(termtype; test_compute=true)
    Ecut=2
    kgrid = [2, 2, 2]
    testcase = silicon
    ε = 1e-8

    xcsym = (termtype isa Xc) ? join(string.(termtype.functionals), " ") : ""
    @testset "Kernel $(typeof(termtype)) $xcsym" begin
        spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
        model = Model(testcase.lattice; atoms=[spec => testcase.positions],
                      terms=[termtype])
        basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
        term  = only(basis.terms)

        ρ0 = guess_density(basis)
        dρ = randn(size(ρ0))

        ρ_minus = ρ0 - ε * dρ
        ρ_plus  = ρ0 + ε * dρ
        pot_minus = DFTK.ene_ops(term, nothing, nothing; ρ=ρ_minus).ops[1].potential
        pot_plus  = DFTK.ene_ops(term, nothing, nothing; ρ=ρ_plus ).ops[1].potential
        dV = (pot_plus - pot_minus) / (2ε)

        dV_apply = DFTK.apply_kernel(term, dρ; ρ=ρ0)
        @test norm(dV - dV_apply) < 100ε

        if test_compute
            kernel = DFTK.compute_kernel(term; ρ=ρ0)
            dV_compute = reshape(kernel * vec(dρ), size(dρ))
            @test norm(dV - dV_compute) < 100ε
        end
    end
end

function test_kernel_collinear_vs_noncollinear(termtype)
    Ecut=2
    kgrid = [2, 2, 2]
    testcase = silicon

    xcsym = (termtype isa Xc) ? join(string.(termtype.functionals), " ") : ""
    @testset "Kernel $(typeof(termtype)) $xcsym (coll == noncoll)" begin
        spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
        model = Model(testcase.lattice; atoms=[spec => testcase.positions],
                      terms=[termtype])
        basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
        term  = only(basis.terms)

        model_col = Model(testcase.lattice; atoms=[spec => testcase.positions],
                          terms=[termtype], spin_polarization=:collinear)
        basis_col = PlaneWaveBasis(model_col, Ecut; kgrid=kgrid)
        term_col  = only(basis_col.terms)

        ρ0 = guess_density(basis)
        dρ = from_real(basis, randn(size(ρ0)))
        dV = DFTK.apply_kernel(term, dρ; ρ=ρ0)[1]

        ρ0     = from_real(basis_col, ρ0.real)
        dρ     = from_real(basis_col, dρ.real)
        ρspin0 = from_real(basis_col, zero(ρ0.real))
        dρspin = from_real(basis_col, zero(ρspin0.real))
        dVα_pol, dVβ_pol = DFTK.apply_kernel(term_col, dρ, dρspin; ρ=ρ0, ρspin=ρspin0)

        @test norm(dVα_pol.real - dVβ_pol.real) < 1e-12
        @test norm(dV.real - dVα_pol.real) < 1e-11
    end
end


## TODO merge with the previous
function test_kernel_collinear(termtype; test_compute=true)
    Ecut=2
    kgrid = [2, 2, 2]
    testcase = silicon
    ε = 1e-8

    xcsym = (termtype isa Xc) ? join(string.(termtype.functionals), " ") : ""
    @testset "Kernel $(typeof(termtype)) $xcsym (collinear)" begin
        spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
        magnetic_moments = [spec => 2rand(2)]
        model = Model(testcase.lattice; atoms=[spec => testcase.positions],
                      terms=[termtype], magnetic_moments=magnetic_moments,
                      spin_polarization=:collinear)
        basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
        iup   = DFTK.krange_spin(basis, 1)[1]  # First spin-up k-point
        idown = DFTK.krange_spin(basis, 2)[1]  # First spin-down k-point
        term  = only(basis.terms)

        ρ0 = guess_density(basis, magnetic_moments)
        dρ = randn(size(ρ0))

        ρ_minus     = ρ0 - ε * dρ
        ρ_plus      = ρ0 + ε * dρ

        ops_minus = DFTK.ene_ops(term, nothing, nothing; ρ=ρ_minus).ops
        ops_plus  = DFTK.ene_ops(term, nothing, nothing; ρ=ρ_plus).ops
        dV = zero(ρ0)
        dV[:, :, :, 1] = (ops_plus[  iup].potential - ops_minus[  iup].potential) / (2ε)
        dV[:, :, :, 2] = (ops_plus[idown].potential - ops_minus[idown].potential) / (2ε)

        dV_apply = DFTK.apply_kernel(term, dρ; ρ=ρ0)
        @test norm(dV - dV_apply) < 200ε

        if test_compute
            kernel = DFTK.compute_kernel(term; ρ=ρ0)
            dV_matrix = reshape(kernel * vec(dρ), size(dρ))
            @test norm(dV - dV_matrix) < 100ε
        end
    end
end

test_kernel_unpolarized(PowerNonlinearity(1.2, 2.0))
test_kernel_unpolarized(Hartree())
test_kernel_unpolarized(Xc(:lda_xc_teter93))
test_kernel_unpolarized(Xc(:gga_c_pbe), test_compute=false)
test_kernel_unpolarized(Xc(:gga_x_pbe), test_compute=false)

test_kernel_collinear_vs_noncollinear(Hartree())
test_kernel_collinear_vs_noncollinear(Xc(:lda_xc_teter93))
test_kernel_collinear_vs_noncollinear(Xc(:gga_c_pbe))
test_kernel_collinear_vs_noncollinear(Xc(:gga_x_pbe))

test_kernel_collinear(Hartree())
test_kernel_collinear(PowerNonlinearity(1.2, 2.5))
test_kernel_collinear(Xc(:lda_xc_teter93))
test_kernel_collinear(Xc(:gga_c_pbe), test_compute=false)
test_kernel_collinear(Xc(:gga_x_pbe), test_compute=false)
test_kernel_collinear(Xc(:gga_x_pbe, :gga_c_pbe), test_compute=false)
