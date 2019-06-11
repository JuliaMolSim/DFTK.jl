lattice = [0.0  5.131570667152971 5.131570667152971;
           5.131570667152971 0.0 5.131570667152971;
           5.131570667152971 5.131570667152971  0.0]

kpoints = [[   0,   0, 0],   # in fractional coordinates
           [ 1/3,   0, 0],
           [ 1/3, 1/3, 0],
           [-1/3, 1/3, 0]]
kweights = [1, 8, 6, 12]
kweights = kweights / sum(kweights)
positions = 1.2828926667882428 * [ones(3), -ones(3)]  # in cartesian coordinates
