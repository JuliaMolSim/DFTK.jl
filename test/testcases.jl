@testmodule TestCases begin
using DFTK
using Unitful
using UnitfulAtomic
using LinearAlgebra: Diagonal, diagm
using LazyArtifacts

hgh_lda_family = artifact"hgh_lda_hgh"
pd_lda_family  = artifact"pd_nc_sr_lda_standard_0.4.1_upf"

silicon = (;
    lattice = [0.0  5.131570667152971 5.131570667152971;
               5.131570667152971 0.0 5.131570667152971;
               5.131570667152971 5.131570667152971  0.0],
    atnum = 14,
    mass = 28.085u"u",
    n_electrons = 8,
    temperature = 0.0,
    psp_hgh = joinpath(hgh_lda_family, "si-q4.hgh"),
    psp_upf = joinpath(pd_lda_family, "Si.upf"),
    positions = [ones(3)/8, -ones(3)/8],      # in fractional coordinates
    kgrid = ExplicitKpoints([[   0,   0, 0],  # kcoords in fractional coordinates
                             [ 1/3,   0, 0],
                             [ 1/3, 1/3, 0],
                             [-1/3, 1/3, 0]],
                            [1/27, 8/27, 6/27, 12/27]),
)
silicon = merge(silicon,
                (; atoms=fill(ElementPsp(silicon.atnum; psp=load_psp(silicon.psp_hgh)), 2)))

magnesium = (;
    lattice = [-3.0179389205999998 -3.0179389205999998 0.0000000000000000;
               -5.2272235447000002 5.2272235447000002 0.0000000000000000;
               0.0000000000000000 0.0000000000000000 -9.7736219469000005],
    atnum = 12,
    mass = 24.305u"u",
    n_electrons = 4,
    psp_hgh = joinpath(hgh_lda_family, "mg-q2.hgh"),
    psp_upf = joinpath(pd_lda_family, "Mg.upf"),
    positions = [[2/3, 1/3, 1/4], [1/3, 2/3, 3/4]],
    kgrid = ExplicitKpoints([[0,   0,   0],
                             [1/3, 0,   0],
                             [1/3, 1/3, 0],
                             [0,   0,   1/3],
                             [1/3, 0,   1/3],
                             [1/3, 1/3, 1/3]],
                            [1/27, 6/27, 2/27, 2/27, 12/27, 4/27]),
    temperature = 0.01,
)
magnesium = merge(magnesium,
                  (; atoms=fill(ElementPsp(magnesium.atnum; psp=load_psp(magnesium.psp_hgh)), 2)))


aluminium = (;
    lattice = Matrix(Diagonal([4 * 7.6324708938577865, 7.6324708938577865,
                               7.6324708938577865])),
    atnum = 13,
    mass = 39.9481u"u",
    n_electrons = 12,
    psp_hgh = joinpath(hgh_lda_family, "al-q3.hgh"),
    psp_upf = joinpath(pd_lda_family, "Al.upf"),
    positions = [[0, 0, 0], [0, 1/2, 1/2], [1/8, 0, 1/2], [1/8, 1/2, 0]],
    temperature = 0.0009500431544769484,
)
aluminium = merge(aluminium,
                  (; atoms=fill(ElementPsp(aluminium.atnum; psp=load_psp(aluminium.psp_hgh)), 4)))


aluminium_primitive = (;
    lattice = [5.39697192863632 2.69848596431816 2.69848596431816;
               0.00000000000000 4.67391479368660 1.55797159787754;
               0.00000000000000 0.00000000000000 4.40660912710674],
    atnum = 13,
    mass = 39.9481u"u",
    n_electrons = 3,
    psp_hgh = joinpath(hgh_lda_family, "al-q3.hgh"),
    psp_upf = joinpath(pd_lda_family, "Al.upf"),
    positions = [zeros(3)],
    temperature = 0.0009500431544769484,
)
aluminium_primitive = merge(aluminium_primitive,
                            (; atoms=fill(ElementPsp(aluminium_primitive.atnum,
                                                     psp=load_psp(aluminium_primitive.psp_hgh)), 1)))


platinum_hcp = (;
    lattice = [10.00000000000000 0.00000000000000 0.00000000000000;
               5.00000000000000 8.66025403784439 0.00000000000000;
               0.00000000000000 0.00000000000000 16.3300000000000],
    atnum = 78,
    mass = 195.0849u"u",
    n_electrons = 36,
    psp_hgh = joinpath(hgh_lda_family, "pt-q18.hgh"),
    psp_upf = joinpath(pd_lda_family, "Pt.upf"),
    positions = [zeros(3), ones(3) / 3],
    temperature = 0.0009500431544769484,
)
platinum_hcp = merge(platinum_hcp,
                     (; atoms=fill(ElementPsp(platinum_hcp.atnum; psp=load_psp(platinum_hcp.psp_hgh)), 2)))

iron_bcc = (;
    lattice = 2.71176 .* [[-1 1 1]; [1 -1  1]; [1 1 -1]],
    atnum = 26,
    mass = 55.8452u"u",
    n_electrons = 8,
    psp_hgh = joinpath(hgh_lda_family, "fe-q8.hgh"),
    psp_upf = joinpath(pd_lda_family, "Fe.upf"),
    positions = [zeros(3)],
    temperature = 0.01,
)
iron_bcc = merge(iron_bcc, (; atoms=[ElementPsp(iron_bcc.atnum; psp=load_psp(iron_bcc.psp_hgh))]))

o2molecule = (;
    lattice = diagm([6.5, 6.5, 9.0]),
    atnum = 8,
    mass = 15.999u"u",
    n_electrons = 12,
    psp_hgh = joinpath(hgh_lda_family, "O-q6.hgh"),
    psp_upf = joinpath(pd_lda_family, "O.upf"),
    positions = 0.1155 * [[0, 0, 1], [0, 0, -1]],
    temperature = 0.02,
)
o2molecule = merge(o2molecule,
                   (; atoms=fill(ElementPsp(o2molecule.atnum; psp=load_psp(o2molecule.psp_hgh)), 2)))

all_testcases = (; silicon, magnesium, aluminium, aluminium_primitive, platinum_hcp,
                 iron_bcc, o2molecule)
end
