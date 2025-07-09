# Atomistic simulation environment (ASE)

The [atomistic simulation environment](https://wiki.fysik.dtu.dk/ase/index.html),
or ASE for short,
is a popular Python package to simplify the process of setting up,
running and analysing results from atomistic simulations across different simulation codes.
By means of [ASEconvert](https://github.com/mfherbst/ASEconvert.jl) this python package
is seamlessly integrated with the AtomsBase ecosystem and thus available to DFTK via
our own [AtomsBase integration](@ref) and [AtomsCalculators integration](@ref).

Most notably this integration is two-way, that is that DFTK can both be used as a
calculator in ASE (to do computation on atomistic systems managed in python) and vice
versa, i.e. atomistic systems set up in ASE can be used as starting points for
calculations in a Julia-driven workflow. Both will be illustrated.

## Using ASE to setup structures for DFTK calculations

An example of this workflow is given in [Modelling a gallium arsenide surface](@ref).

## Using DFTK as a calculator in ASE

!!! warning "Recent feature"
    This is a relatively new feature and still has rough edges.
    We appreciate any feedback, bug reports or PRs.

The [IPICalculator](https://github.com/JuliaMolSim/IPICalculator.jl)
package contains an implementation of the [i-PI protocol](https://github.com/i-pi/i-pi)
to pass energies, forces and virials between different atomistic calculators.
This can be used, for example, to make julia-based calculators available in ASE.

The following example requests the computation of a silicon energy and force
from DFTK.

```python
import ase.build
from ase.calculators.socketio import SocketIOCalculator

atoms = ase.build.bulk("Si")
atoms.write("Si.cif")
atoms.calc = SocketIOCalculator(unixsocket="dftk")
atoms.get_forces()  # Stalls until you connect a driver
```

To supply this force request with an PBE force, we need to setup
an initial system as a template as well as an PBE calculator
with desired numerical parameters:

```julia
using AtomsIO
using DFTK
using IPICalculator
using PseudoPotentialData

# Setup DFTK calculator
calc = DFTKCalculator(
    model_kwargs=(; pseudopotentials=PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf"),
                    functionals=PBE(), temperature=1e-3, smearing=Smearing.Gaussian()),
    basis_kwargs=(; Ecut=20, kgrid=(4, 4, 4)),
)

# Run IPI driver
run_driver(load_system("Si.cif"), calc; unixsocket="dftk")
```
