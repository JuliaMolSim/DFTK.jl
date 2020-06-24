using Test
using DFTK

function load_reference(folder::EtsfFolder)
    n_kpoints = size(folder.gsr["reduced_coordinates_of_kpoints"], 2)
    bands = Vector{Vector{Float64}}(undef, n_kpoints)
    for ik in 1:n_kpoints
        bands[ik] = Vector(folder.eig["Eigenvalues"][:, ik, 1])
    end

    energies = Dict{String, Float64}(
         "Entropy"        => folder.gsr["e_entropy"][:],
         "Ewald"          => folder.gsr["e_ewald"][:],
         "PspCorrection"  => folder.gsr["e_corepsp"][:],
         "Xc"             => folder.gsr["e_xc"][:],
         "Kinetic"        => folder.gsr["e_kinetic"][:],
         "Hartree"        => folder.gsr["e_hartree"][:],
         "AtomicLocal"    => folder.gsr["e_localpsp"][:],
         "AtomicNonlocal" => folder.gsr["e_nonlocalpsp"][:],
    )

    (energies=energies, bands=bands)
end


function test_folder(T, folder; scf_tol=1e-6, n_ignored=0, test_tol=1e-6)
    @testset "$folder" begin
        etsf = EtsfFolder(folder)

        basis = load_basis(T, etsf)
        atoms = load_atoms(T, etsf)
        ref = load_reference(etsf)
        n_bands = length(ref.bands[1])

        is_converged = DFTK.scf_convergence_density_difference(scf_tol)
        scfres = self_consistent_field(basis, n_bands=n_bands, is_converged=is_converged)
        println("etot    ", sum(scfres.energies) - sum(ref.energies))
        for ik in 1:length(basis.kpoints)
            @test eltype(scfres.eigenvalues[ik]) == T
            @test eltype(scfres.ψ[ik]) == Complex{T}
            println(ik, "  ", abs.(scfres.eigenvalues[ik][1:n_bands] - ref.bands[ik]))
        end
        for ik in 1:length(basis.kpoints)
            # Ignore last few bands, because these eigenvalues are hardest to converge
            # and typically a bit random and unstable in the LOBPCG
            diff = abs.(scfres.eigenvalues[ik][1:n_bands] - ref.bands[ik])
            @test maximum(diff[1:n_bands - n_ignored]) < test_tol
        end
        for (key, value) in pairs(scfres.energies)
            if haskey(ref.energies, key)
                @test value ≈ ref.energies[key] atol=5test_tol
            else
                @warn "Key not tested: $key"
            end
        end
        @test sum(scfres.energies) ≈ sum(ref.energies) atol=test_tol
    end
end


function main()
    test_folder(Float64, "silicon_E10_k3_LDA", test_tol=5e-8)
    test_folder(Float64, "silicon_E15_k4_LDA", n_ignored=1, test_tol=5e-8)
    test_folder(Float64, "silicon_E25_k3_LDA", n_ignored=3, test_tol=1e-8)
    test_folder(Float64, "silicon_E15_k4_GGA")
    test_folder(Float64, "silicon_E25_k3_GGA")
    test_folder(Float64, "silicon_E25_k4_GGA", n_ignored=1, test_tol=1e-7)
    test_folder(Float64, "magnesium_E15_k3_LDA_Fermi_Dirac", n_ignored=2)
    test_folder(Float64, "magnesium_E15_k3_GGA_Methfessel_Paxton", n_ignored=1)
    test_folder(Float64, "magnesium_E25_k5_GGA_Methfessel_Paxton", n_ignored=2)
    test_folder(Float64, "graphite_E20_k8_LDA_Methfessel_Paxton", scf_tol=7e-5,
                test_tol=7e-4, n_ignored=2)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

