include("../testcases.jl")

structure = build_silicon_structure()
pspfile = joinpath(@__DIR__, "Si-q4-pade.abinit.hgh")

infile = abilab.AbinitInput(structure=structure, pseudos=abidata.pseudos(pspfile))
infile.set_kmesh(ngkpt=[4, 4, 4], shiftk=[0, 0, 0])
infile.set_vars(
    ecut=15,        # Hartree
    nband=6,        # Number of bands
    tolvrs=1e-10,   # General tolerance settings
    ixc=1           # LDA_XC_TETER93
)
infile.extra = Dict("pspmap" => Dict("Si" => "si-pade-q4.hgh", ), )

run_ABINIT_scf(infile, @__DIR__)
