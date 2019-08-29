include("../testcases.jl")

structure = build_graphite_structure()
pspfile = joinpath(abidata.__path__[end], "hgh_pseudos/6c.4.hgh")

infile = abilab.AbinitInput(structure=structure, pseudos=abidata.pseudos(pspfile))
infile.set_kmesh(ngkpt=[8, 8, 8], shiftk=[0, 0, 0])
infile.set_vars(
    ecut=20,        # Hartree
    nband=12,       # Number of bands
    tolvrs=1e-10,   # General tolerance settings
    ixc=1,          # Teter LDA
    occopt=6,       # Methfessel and Paxton, Hermite polynomial degree 2
    tsmear=0.01,    # Hartree
)
infile.extra = Dict("pspmap" => Dict("C" => "c-pade-q4.hgh", ), )

run_ABINIT_scf(infile, @__DIR__)
