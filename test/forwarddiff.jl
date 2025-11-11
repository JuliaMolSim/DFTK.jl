### ForwardDiff tests on the CPU
@testitem "Force derivatives using ForwardDiff" #=
    =#    tags=[:dont_test_mpi, :minimal] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    ForwardDiffCases.test_FD_force_derivatives(DFTK.CPU())
end

@testitem "Strain sensitivity using ForwardDiff" #=
    =#    tags=[:dont_test_mpi, :minimal] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    ForwardDiffCases.test_FD_strain_sensitivity(DFTK.CPU())
end

@testitem "scfres PSP sensitivity using ForwardDiff" #=
    =#    tags=[:dont_test_mpi, :minimal] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    ForwardDiffCases.test_FD_psp_sensitivity(DFTK.CPU())
end

@testitem "Functional force sensitivity using ForwardDiff" #=
    =#    tags=[:dont_test_mpi, :minimal] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    ForwardDiffCases.test_FD_force_sensitivity(DFTK.CPU())
end

@testitem "LocalNonlinearity sensitivity using ForwardDiff" #=
    =#    tags=[:dont_test_mpi, :minimal] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    ForwardDiffCases.test_FD_local_nonlinearity_sensitivity(DFTK.CPU())
end

@testitem "Symmetries broken by perturbation are filtered out" #=
    =# tags=[:dont_test_mpi, :minimal] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    ForwardDiffCases.test_FD_filter_broken_symmetries(DFTK.CPU())
end

@testitem "Symmetry-breaking perturbation using ForwardDiff" #=
    =# tags=[:dont_test_mpi, :minimal] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    ForwardDiffCases.test_FD_symmetry_breaking_perturbation(DFTK.CPU())
end

@testitem "Test scfres dual has the same params as scfres primal" #=
    =# tags=[:dont_test_mpi, :minimal] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    ForwardDiffCases.test_FD_scfres_parameter_consistency(DFTK.CPU())
end

@testitem "ForwardDiff wrt temperature" #=
    =# tags=[:dont_test_mpi, :minimal] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    ForwardDiffCases.test_FD_wrt_temperature(DFTK.CPU())
end

### Generic ForwardDiff tests (independent of DFTK architecture)
@testitem "Derivative of complex function" #=
    =# tags=[:dont_test_mpi, :minimal] setup=[TestCases, ForwardDiffCases] begin
    ForwardDiffCases.test_FD_complex_function_derivative()
end

@testitem "Higher derivatives of Fermi-Dirac occupation" #=
    =# tags=[:dont_test_mpi, :minimal] setup=[TestCases, ForwardDiffCases] begin
    ForwardDiffCases.test_FD_fermi_dirac_higher_derivatives()
end

