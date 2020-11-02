using Plots

L = 5  # Length of axis (bohr)
w = 1  # Width of dipole layer (bohr)

wfrac = w / L  # width in fractional coordinates
whalf = wfrac / 2  # to each side
@assert  whalf < 1/2

N = 100000
x  = Array(0:(1/N):1)  # fractional coordinates

sawtooth = @. x - 0.5;
sawtooth_orig = copy(sawtooth)

a = 1 - 3 / 4whalf
b = 1 / (4whalf^3)

gc = ceil(Int, N * whalf)
@. sawtooth[1:gc] = x[1:gc] * (a + b * x[1:gc]^2)
sawtooth[end-gc+1:end] = -reverse(sawtooth[1:gc])


plot(x, sawtooth_orig)
plot!(x, sawtooth)
ylims!(-2, 2)
