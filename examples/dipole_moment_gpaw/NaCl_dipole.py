from gpaw import GPAW, PW
from ase.io import read

nacl = read("../NaCl.in")
nacl.calc = GPAW(txt='corrected.txt',
                 xc='PBE',
                 kpts=(1, 1, 1),
                 mode=PW(272.11386020632835),
                 )
nacl.pbc = (True, True, True)
e3 = nacl.get_potential_energy()
nacl.calc.write('nacl.gpw')
