include("../testcases.jl")

structure = build_magnesium_structure()
pspfile = joinpath(abidata.__path__[end], "hgh_pseudos/12mg.2.hgh")

infile = abilab.AbinitInput(structure=structure, pseudos=abidata.pseudos(pspfile))
infile.set_kmesh(ngkpt=[5, 5, 5], shiftk=[0, 0, 0])
infile.set_vars(
    ecut=25,        # Hartree
    nband=10,       # Number of bands
    tolvrs=1e-10,   # General tolerance settings
    ixc="-101130",  # PBE X and PBE C
    occopt=6,       # Methfessel and Paxton, Hermite polynomial degree 2
    tsmear=0.01,    # Hartree
)
infile.extra = Dict("pspmap" => Dict("Mg" => "mg-pade-q2.hgh", ), )

run_ABINIT_scf(infile, @__DIR__)
