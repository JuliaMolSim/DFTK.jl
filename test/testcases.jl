using LinearAlgebra

silicon = (
    lattice = [0.0  5.131570667152971 5.131570667152971;
               5.131570667152971 0.0 5.131570667152971;
               5.131570667152971 5.131570667152971  0.0],
    atnum = 14,
    n_electrons = 8,
    temperature = nothing,
    psp = "hgh/lda/si-q4",
    positions = [ones(3)/8, -ones(3)/8],  # in fractional coordinates
    kcoords = [[   0,   0, 0],  # in fractional coordinates
               [ 1/3,   0, 0],
               [ 1/3, 1/3, 0],
               [-1/3, 1/3, 0]],
    kweights = [1/27, 8/27, 6/27, 12/27],
)
silicon = merge(silicon,
                (; atoms=fill(ElementPsp(silicon.atnum, psp=load_psp(silicon.psp)), 2)))

magnesium = (
    lattice = [-3.0179389205999998 -3.0179389205999998 0.0000000000000000;
               -5.2272235447000002 5.2272235447000002 0.0000000000000000;
               0.0000000000000000 0.0000000000000000 -9.7736219469000005],
    atnum = 12,
    n_electrons = 4,
    psp = "hgh/lda/mg-q2",
    positions = [[2/3, 1/3, 1/4], [1/3, 2/3, 3/4]],
    kcoords =  [[0,   0,   0],
                [1/3, 0,   0],
                [1/3, 1/3, 0],
                [0,   0,   1/3],
                [1/3, 0,   1/3],
                [1/3, 1/3, 1/3]],
    temperature = 0.01,
    kweights = [1/27, 6/27, 2/27, 2/27, 12/27, 4/27],
)
magnesium = merge(magnesium,
                  (; atoms=fill(ElementPsp(magnesium.atnum, psp=load_psp(magnesium.psp)), 2)))


aluminium = (
    lattice = Matrix(Diagonal([4 * 7.6324708938577865, 7.6324708938577865,
                               7.6324708938577865])),
    atnum = 13,
    n_electrons = 12,
    psp = "hgh/lda/al-q3",
    positions = [[0, 0, 0], [0, 1/2, 1/2], [1/8, 0, 1/2], [1/8, 1/2, 0]],
    temperature = 0.0009500431544769484,
)
aluminium = merge(aluminium,
                  (; atoms=fill(ElementPsp(aluminium.atnum, psp=load_psp(aluminium.psp)), 4)))


aluminium_primitive = (
    lattice = [5.39697192863632 2.69848596431816 2.69848596431816;
               0.00000000000000 4.67391479368660 1.55797159787754;
               0.00000000000000 0.00000000000000 4.40660912710674],
    atnum = 13,
    n_electrons = 3,
    psp = "hgh/lda/al-q3",
    positions = [[0, 0, 0]],
    temperature = 0.0009500431544769484,
)


platinum_hcp = (
    lattice = [10.00000000000000 0.00000000000000 0.00000000000000;
               5.00000000000000 8.66025403784439 0.00000000000000;
               0.00000000000000 0.00000000000000 16.3300000000000],
    atnum = 78,
    n_electrons = 36,
    psp = "hgh/lda/pt-q18",
    positions = [zeros(3), ones(3) / 3],
    temperature = 0.0009500431544769484,
)
platinum_hcp = merge(platinum_hcp,
                     (; atoms=fill(ElementPsp(platinum_hcp.atnum, psp=load_psp(platinum_hcp.psp)), 2)))

iron_bcc = (
    lattice = 2.71176 .* [[-1 1 1]; [1 -1  1]; [1 1 -1]],
    atnum = 26,
    n_electrons = 8,
    psp = "hgh/lda/fe-q8",
    positions = [zeros(3)],
    temperature = 0.01,
)
iron_bcc = merge(iron_bcc, (; atoms=[ElementPsp(iron_bcc.atnum, psp=load_psp(iron_bcc.psp))]))

o2molecule = (
    lattice = diagm([6.5, 6.5, 9.0]),
    atnum = 8,
    n_electrons = 12,
    psp = "hgh/lda/O-q6",
    positions = 0.1155 * [[0, 0, 1], [0, 0, -1]],
    temperature = 0.02,
)
