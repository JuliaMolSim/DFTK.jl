include("../testcases.jl")

structure = build_magnesium_structure()
pspfile = joinpath(abidata.__path__[end], "hgh_pseudos/12mg.2.hgh")

infile = abilab.AbinitInput(structure=structure, pseudos=abidata.pseudos(pspfile))
infile.set_kmesh(ngkpt=[3, 3, 3], shiftk=[0, 0, 0])
infile.set_vars(
    ecut=15,        # Hartree
    nband=10,       # Number of bands
    tolvrs=1e-10,   # General tolerance settings
    ixc="-020",     # Teter1993 LDA reparametrisation
    # occopt=6,     # Methfessel and Paxton, Hermite polynomial degree 2
    occopt=3,       # Fermi-Dirac
    tsmear=0.01,    # Hartree
)
infile.extra = Dict("pspmap" => Dict(12 => "hgh/lda/mg-q2", ), )

run_ABINIT_scf(infile, @__DIR__)
