using DFTK
using Test

include("testcases.jl")

function test_kernel(spin_polarization, termtype; test_compute=true)
    kgrid = [2, 2, 2]
    testcase = silicon
    ε   = 1e-8
    tol = 1e-5

    xcsym = (termtype isa Xc) ? join(string.(termtype.functionals), " ") : ""
    @testset "Kernel $(typeof(termtype)) $xcsym ($spin_polarization)" begin
        spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
        magnetic_moments = []
        n_spin = 1
        if spin_polarization == :collinear
            magnetic_moments = [spec => 2rand(2)]
            n_spin = 2
        end

        model = Model(testcase.lattice; atoms=[spec => testcase.positions],
                      terms=[termtype], magnetic_moments=magnetic_moments,
                      spin_polarization=spin_polarization)
        @test model.n_spin_components == n_spin
        basis = PlaneWaveBasis(model; Ecut=2, kgrid=kgrid)
        term  = only(basis.terms)

        ρ0 = guess_density(basis, magnetic_moments)
        δρ = randn(size(ρ0))
        ρ_minus     = ρ0 - ε * δρ
        ρ_plus      = ρ0 + ε * δρ
        ops_minus = DFTK.ene_ops(term, nothing, nothing; ρ=ρ_minus).ops
        ops_plus  = DFTK.ene_ops(term, nothing, nothing; ρ=ρ_plus).ops
        δV = zero(ρ0)

        for iσ in 1:model.n_spin_components
            # Index of the first spin-up or spin-down k-point
            ifirst = first(DFTK.krange_spin(basis, iσ))
            δV[:, :, :, iσ] = (ops_plus[ifirst].potential - ops_minus[ifirst].potential) / (2ε)
        end

        δV_apply = DFTK.apply_kernel(term, δρ; ρ=ρ0)
        @test norm(δV - δV_apply) < tol
        if test_compute
            kernel = DFTK.compute_kernel(term; ρ=ρ0)
            δV_matrix = reshape(kernel * vec(δρ), size(δρ))
            @test norm(δV - δV_matrix) < tol
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
        basis = PlaneWaveBasis(model; Ecut, kgrid)
        term  = only(basis.terms)

        model_col = Model(testcase.lattice; atoms=[spec => testcase.positions],
                          terms=[termtype], spin_polarization=:collinear)
        basis_col = PlaneWaveBasis(model_col; Ecut, kgrid)
        term_col  = only(basis_col.terms)

        ρ0 = guess_density(basis)
        δρ = randn(size(ρ0))
        δV = DFTK.apply_kernel(term, δρ; ρ=ρ0)

        ρ0_col = cat(0.5ρ0, 0.5ρ0, dims=4)
        δρ_col = cat(0.5δρ, 0.5δρ, dims=4)
        δV_pol = DFTK.apply_kernel(term_col, δρ_col; ρ=ρ0_col)

        @test norm(δV_pol[:, :, :, 1] - δV_pol[:, :, :, 2]) < 1e-12
        @test norm(δV - δV_pol[:, :, :, 1:1]) < 1e-11
    end
end

test_kernel(:none, PowerNonlinearity(1.2, 2.0))
test_kernel(:none, Hartree())
test_kernel(:none, Xc(:lda_xc_teter93))
test_kernel(:none, Xc(:gga_c_pbe), test_compute=false)
test_kernel(:none, Xc(:gga_x_pbe), test_compute=false)

test_kernel_collinear_vs_noncollinear(Hartree())
test_kernel_collinear_vs_noncollinear(Xc(:lda_xc_teter93))
test_kernel_collinear_vs_noncollinear(Xc(:gga_c_pbe))
test_kernel_collinear_vs_noncollinear(Xc(:gga_x_pbe))

test_kernel(:collinear, Hartree())
test_kernel(:collinear, PowerNonlinearity(1.2, 2.5))
test_kernel(:collinear, Xc(:lda_xc_teter93))
test_kernel(:collinear, Xc(:gga_c_pbe), test_compute=false)
test_kernel(:collinear, Xc(:gga_x_pbe), test_compute=false)
test_kernel(:collinear, Xc(:gga_x_pbe, :gga_c_pbe), test_compute=false)
