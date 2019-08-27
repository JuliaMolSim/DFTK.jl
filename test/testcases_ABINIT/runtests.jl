include("testcases.jl")

test_folder(Float64, "silicon_E10_k3_LDA", test_tol=5e-8)
test_folder(Float64, "silicon_E15_k4_LDA", n_ignored=1, test_tol=5e-8)
test_folder(Float64, "silicon_E25_k3_LDA", n_ignored=2, test_tol=1e-8)
test_folder(Float64, "silicon_E15_k4_GGA")
test_folder(Float64, "silicon_E25_k3_GGA")
test_folder(Float64, "silicon_E25_k4_GGA", n_ignored=1, test_tol=1e-7)
test_folder(Float64, "magnesium_E15_k3_LDA_Fermi_Dirac", n_ignored=2)
test_folder(Float64, "magnesium_E15_k3_GGA_Methfessel_Paxton", n_ignored=1)
test_folder(Float64, "magnesium_E25_k5_GGA_Methfessel_Paxton", n_ignored=2)
test_folder(Float64, "graphite_E20_k8_LDA_Methfessel_Paxton", scf_tol=7e-5, test_tol=7e-4,
            n_ignored=2)

# for file in readdir()
#     if isdir(file) && isfile(joinpath(file, "generate.jl"))
#         @testset "$file" begin
#             if isfile(joinpath(file, "extra.jld")) && isfile(joinpath(file, "out_GSR.nc"))
#                 test_folder(Float64, file)
#             end
#         end
#         run_folder(file)
#     end
# end
