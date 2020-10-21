using DFTK
using Test
using LinearAlgebra: norm

include("testcases.jl")

function test_chi0(;symmetry=false, use_symmetry=false, temperature=0,
                   spin_polarization=:none,
                   kgrid=[3, 1, 1], fft_size=[10, 1, 10], Ecut=3)
    testcase = silicon
    n_bands  = 12
    tol      = 1e-14
    ε        = 1e-8
    testtol  = 2e-6

    collinear = spin_polarization == :collinear
    label = [
        symmetry        ? "   symm" : "no symm",
        use_symmetry    ? "   use" : "no use",
        temperature > 0 ? "temp" : "  0K",
        collinear       ? "coll" : "none",
    ]
    @testset "Computing χ0 ($(join(label, ", ")))" begin
        spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
        magnetic_moments = collinear ? [spec => 2rand(2)] : []
        model_kwargs = (temperature=temperature, symmetries=symmetry,
                        magnetic_moments=magnetic_moments,
                        spin_polarization=spin_polarization)
        basis_kwargs = (kgrid=kgrid, fft_size=fft_size, use_symmetry=use_symmetry)
        model = model_LDA(testcase.lattice, [spec => testcase.positions]; model_kwargs...)
        basis = PlaneWaveBasis(model, Ecut; basis_kwargs...)

        ρ0     = guess_density(basis)
        ρspin0 = guess_spin_density(basis, magnetic_moments)
        energies, ham0 = energy_hamiltonian(basis, nothing, nothing; ρ=ρ0, ρspin=ρspin0)
        res = DFTK.next_density(ham0, tol=tol, eigensolver=diag_full, n_bands=n_bands)
        ρ1  = res.ρout
        ρspin1 = res.ρ_spin_out

        # Now we make the same model, but add an artificial external potential ε * dV
        n_spin = model.n_spin_components
        dV = [from_real(basis, randn(eltype(basis), basis.fft_size)) for _ in 1:n_spin]
        dV_sym = DFTK.symmetrize.(dV)
        if symmetry
            dV = dV_sym
        else
            for σ in 1:n_spin
                @test dV_sym[σ].real ≈ dV[σ].real
            end
        end

        εdV = cat((ε .* dV[σ].real for σ in 1:n_spin)...; dims=4)
        term_builder = basis -> DFTK.TermExternal(basis, εdV)
        model = model_LDA(testcase.lattice, [spec => testcase.positions];
                          model_kwargs..., extra_terms=[term_builder])
        basis = PlaneWaveBasis(model, Ecut; basis_kwargs...)
        energies, ham = energy_hamiltonian(basis, nothing, nothing; ρ=ρ0, ρspin=ρspin0)
        res = DFTK.next_density(ham, tol=tol, eigensolver=diag_full, n_bands=n_bands)
        ρ2     = res.ρout
        ρspin2 = res.ρ_spin_out
        diff_findiff = [(ρ2 - ρ1) / ε]
        !isnothing(ρspin1) && push!(diff_findiff, (ρspin2 - ρspin1) / ε)

        EVs = [eigen(Hermitian(Array(Hk))) for Hk in ham0.blocks]
        Es = [EV.values[1:n_bands] for EV in EVs]
        Vs = [EV.vectors[:, 1:n_bands] for EV in EVs]
        occ, εF = find_occupation(basis, Es)

        # Test apply_χ0 and compare against finite differences
        diff_applied_χ0 = apply_χ0(ham0, Vs, εF, Es, dV...; droptol=0)
        for σ in 1:n_spin
            @test norm(diff_findiff[σ].real - diff_applied_χ0[σ].real) < testtol
        end

        if !symmetry
            dV_vec = vcat(vec.(getproperty.(dV, :real))...)

            # Test compute_χ0 against finite differences
            χ0 = compute_χ0(ham0)
            diff_computed_χ0 = real(reshape(χ0 * dV_vec, basis.fft_size..., n_spin))
            for σ in 1:n_spin
                @test norm(diff_findiff[σ].real - diff_computed_χ0[:, :, :, σ]) < testtol
            end

            # Test that apply_χ0 is self-adjoint
            dV1 = [from_real(basis, randn(eltype(basis), basis.fft_size)) for _ in 1:n_spin]
            dV2 = [from_real(basis, randn(eltype(basis), basis.fft_size)) for _ in 1:n_spin]
            χ0dV1 = apply_χ0(ham0, Vs, εF, Es, dV1...)
            χ0dV2 = apply_χ0(ham0, Vs, εF, Es, dV2...)
            if n_spin == 1
                @test abs(  dot(dV1[1].real, χ0dV2[1].real)
                          - dot(dV2[1].real, χ0dV1[1].real)) < testtol
            else
                # Form linear combination to get density changes in α and β
                χ0dV1αβ = [(χ0dV1[1].real + χ0dV1[2].real) / 2,  # (ρtot + ρspin) / 2
                           (χ0dV1[1].real - χ0dV1[2].real) / 2]  # (ρtot - ρspin) / 2
                χ0dV2αβ = [(χ0dV2[1].real + χ0dV2[2].real) / 2,
                           (χ0dV2[1].real - χ0dV2[2].real) / 2]
                V1χ0V2 = dot(χ0dV1αβ[1], dV2[1].real) + dot(χ0dV1αβ[2], dV2[2].real)
                V2χ0V1 = dot(χ0dV2αβ[1], dV1[1].real) + dot(χ0dV2αβ[2], dV1[2].real)
                @test abs(V1χ0V2 - V2χ0V1) < testtol
            end

            # Test the diagonal_only option
            χ0_diag = compute_χ0(ham0; droptol=Inf)
            diff_diag_1 = real(reshape(χ0_diag * dV_vec, basis.fft_size..., n_spin))
            diff_diag_2 = apply_χ0(ham0, Vs, εF, Es, dV...; droptol=Inf,
                                   sternheimer_contribution=false)
            for σ in 1:n_spin
                @test norm(diff_diag_1[:, :, :, σ] - diff_diag_2[σ].real) < testtol
            end
        end
    end
end

for temperature in (0, 0.03), spin_polarization in (:none, :collinear)
    for use_symmetry in (false, true), symmetry in (false, true)
        test_chi0(;symmetry=symmetry, use_symmetry=use_symmetry,
                  temperature=temperature, spin_polarization=spin_polarization)
    end
end
