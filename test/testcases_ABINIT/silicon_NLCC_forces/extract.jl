using NCDatasets

gsr = NCDataset("./silicon_NLCC_forceso_GSR.nc")
ref_energies = Dict{String, Float64}(
    "Entropy"        => gsr["e_entropy"][:],
    "Ewald"          => gsr["e_ewald"][:],
    "PspCorrection"  => gsr["e_corepsp"][:],
    "Xc"             => gsr["e_xc"][:],
    "Kinetic"        => gsr["e_kinetic"][:],
    "Hartree"        => gsr["e_hartree"][:],
    "AtomicLocal"    => gsr["e_localpsp"][:],
    "AtomicNonlocal" => gsr["e_nlpsp_vfock"][:],
)
ref_evals_ABINIT = gsr["eigenvalues"][:,:,1]
ref_evals_ABINIT = [col for col in eachcol(ref_evals_ABINIT)]
ref_ABINIT = gsr["etotal"][:]
ref_fermi = gsr["fermie"][:]

@show ref_energies
@show ref_evals_ABINIT
@show ref_fermi
@show ref_ABINIT
