include("run_scf_and_compare.jl")
include("testcases.jl")
using DoubleFloats
using GenericLinearAlgebra

# Silicon redHF (without xc) is a metal, so we add a bit of temperature to it
function run_silicon_redHF(T; Ecut=5, grid_size=15, spin_polarization=:none, kwargs...)
    # T + Vloc + Vnloc + Vhartree
    # These values were computed using ABINIT with the same kpoints as testcases.jl
    # and Ecut = 25
    ref_redHF = [
        [0.17899118507651615, 0.6327279881297371, 0.6327279881326648, 0.6327279881356039,
         0.706557757783828, 0.7065577577877139, 0.7065577577915956, 0.7397951816714727,
         0.8532089291297222, 0.8978914445971602],
        [0.23220003663858457, 0.42189409862896016, 0.5921574659414509, 0.5921574659446628,
         0.672858189872362, 0.7372271903827399, 0.7372271903861028, 0.8643640848936627,
         0.9011792204214196, 0.9011792204356576],
        [0.2517502116803524, 0.445206025448218, 0.5328870916963034, 0.532887091701182,
         0.6211365856991057, 0.661989858948651, 0.8863951918546257, 0.8863951918584175,
         0.973261179805555, 0.9771287508158364],
        [0.30685586314464863, 0.376375429632464, 0.4438764716222098, 0.5459065154292047,
         0.651122698647485, 0.8164293660861612, 0.8515978828421051, 0.8735213568005982,
         0.8807275612483988, 0.8886454931307763]
    ]
    ref_etot = -5.440593269861395

    fft_size = fill(grid_size, 3)
    fft_size = DFTK.next_working_fft_size(T, fft_size) # ad-hoc fix for buggy generic FFTs
    Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.atnum, functional="lda", family="hgh"))
    model = model_DFT(Array{T}(silicon.lattice), [Si => silicon.positions], [];
                      temperature=0.05, spin_polarization=spin_polarization)
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)

    spin_polarization == :collinear && (ref_redHF = vcat(ref_redHF, ref_redHF))
    run_scf_and_compare(T, basis, ref_redHF, ref_etot;
                        ρ=guess_density(basis),
                        kwargs...)
end


@testset "Silicon without XC (small)" begin
    run_silicon_redHF(Float64, Ecut=5, test_tol=0.05, n_ignored=0, grid_size=15,
                      test_etot=false)
end

if !isdefined(Main, :FAST_TESTS) || !FAST_TESTS
    @testset "Silicon without XC (large)" begin
        run_silicon_redHF(Float64, Ecut=25, test_tol=1e-5, n_ignored=2, grid_size=35,
                          scf_tol=1e-7, test_etot=false)
    end
end

# There is a weird race condition with MPI + DoubleFloats. TODO debug
if mpi_nprocs() == 1
    @testset "Silicon without XC (small, Double32)" begin
        run_silicon_redHF(Double32, Ecut=5, test_tol=0.05, n_ignored=0, grid_size=15,
                          test_etot=false)
    end
end


@testset "Silicon without XC (small, collinear spin)" begin
    run_silicon_redHF(Float64, Ecut=5, test_tol=0.05, n_ignored=0, grid_size=15,
                      test_etot=false, spin_polarization=:collinear)
end
