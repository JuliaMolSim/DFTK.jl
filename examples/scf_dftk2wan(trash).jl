using DFTK

a = 10.26
lattice = a / 2*[[-1.  0. -1.];
                 [ 0   1.  1.];
                 [ 1   1.  0.]]

Si = ElementPsp(:Si, psp=load_psp("hgh/pbe/Si-q4"))
atoms = [ Si => [zeros(3), 0.25*[-1,3,-1]] ]

model = model_PBE(lattice,atoms)

kgrid = [4,4,4]
Ecut = 20
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, optimize_fft_size = true, use_symmetry=false)

scfres = self_consistent_field(basis, tol=1e-12, n_bands = 12, n_ep_extra = 0);
# scfres = self_consistent_field(basis, tol=1e-12, n_bands = 4, n_ep_extra = 0);

# dftk2wan_win_file("Si", basis, scfres, kgrid, 8;
#                   bands_plot=false, num_print_cycles=50, num_iter=500,
#                   dis_win_max       = "17.185257d0",
#                   dis_froz_max      =  "6.8318033d0",
#                   dis_num_iter      =  120,
#                   dis_mix_ratio     = "1d0")

# gaussian_centers = [[0.09611870429526492 0.5235290324551867 0.1482727073029615],
#                     [0.3042596500813055 0.39096307835453037 0.09817406622820646],
#                     [0.831047015966353 0.8279776870969524 0.3779108971449747],
#                     [0.30602604557771507 0.6093721290521086 0.7591324705458549],
#                     [0.3354398946220738 0.22603890755477218 0.9723396405804758],
#                     [0.22346216793907248 0.01517530607967732 0.3298288543989589],
#                     [0.4363025633802393 0.821779555205526 0.9374451887917208],
#                     [0.7639149773638563 0.07851620747046106 0.33835828273919355]]


# dftk2wan_wannierization_files("Si", basis, scfres, 8;
#                               guess = "gaussian",
#                               centers = gaussian_centers, coords = "reduced")
