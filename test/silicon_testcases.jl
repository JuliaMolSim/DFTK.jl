lattice = [0.0  5.131570667152971 5.131570667152971;
           5.131570667152971 0.0 5.131570667152971;
           5.131570667152971 5.131570667152971  0.0]
kpoints = [[   0,   0, 0],   # in fractional coordinates
           [ 1/3,   0, 0],
           [ 1/3, 1/3, 0],
           [-1/3, 1/3, 0]]
kweights = [1, 8, 6, 12]
kweights = kweights / sum(kweights)
positions = [ones(3)/8, -ones(3)/8]  # in fractional coordinates
