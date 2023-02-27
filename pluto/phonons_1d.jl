### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 1d1035e2-a2d9-4879-ae02-01775731fb61
begin
	import Pkg
	Pkg.add(url="https://github.com/epolack/DFTK.jl", rev="phonon")
	# using Revise
	# Pkg.develop(path="/home/maths/worktree/DFTK.jl/phonon_example_1d")
end;

# ╔═╡ b53cf69c-3cf8-44bf-8fef-5483ab1b382d
begin
	using DFTK
	using LinearAlgebra
	using PyPlot
	using PlutoUI  # Can't plot in a loop…
	using LaTeXStrings
end

# ╔═╡ 2d4bea83-e31c-424f-b73e-30c0b7e97ce7
md"""
# Example of phonon bands computations for a one-dimensional periodic system.
"""

# ╔═╡ a24e6313-0e18-4e05-99ce-8ca6133fde33
md"""
In this example, we look at how to get phonon bands with for a toy system of nuclei
generating a Gaussian potential and which interacts with a Lennard–Jones alike potential.
"""

# ╔═╡ b7594e68-a789-433f-acc8-c0bc9ba432a2
TableOfContents()

# ╔═╡ e4b5ebfa-ff93-4945-888a-fa8e69d80aa8
md"""
## User parameters

The example system consists of `n_atoms` atoms.

It is also possible to modify the `εF` Fermi level. For example, a Fermi level of `-1.22` will only deal with one occupied orbital for the parametrs we will set below.
This can be seen by looking at `scfres.occupation` and `scfres.eigenvalues`.

By default, with `εF` set to `nothing`, we will look for `n_atoms` occupied orbitals.
"""

# ╔═╡ 01debbd5-357a-4494-8586-3ff98883dd34
begin
	n_atoms = 2
	εF = nothing
end;

# ╔═╡ bbae56f8-0e3d-4497-97cd-704f5d2c6124
md"""
## Setting-up the system
"""

# ╔═╡ f741a53c-7bb6-4f80-bdf6-551aeef1fec7
md"""
We create a `prepare_unit_cell` function that takes the number of unit cell atoms we want to model and the modes associated to them (`+` or `-` for a one-dimensional system).

In phonon computations, we are interested in normal modes. That is discrete collective variables obtained from diagonalising the second-derivative of the energy (the Hessian) for displacements of the atoms in the infinite system.

This collective movements of the atoms can for example explain the speed of propagation of sound waves in materials.
"""

# ╔═╡ 6dc3d7de-c435-4092-8057-5721188a71db
md"""
### The unit cell
"""

# ╔═╡ 8b3215b4-8e02-40dc-83d3-95521c21fbdc
function prepare_unit_cell(n_atoms=2, modes=[:+ for _ in 1:n_atoms]; hpot=3.0, spread=0.5, do_hartree=false, do_nonlinear=false)
	@assert length(modes) == n_atoms
	a = 5.0
	lattice = a * n_atoms .* [1.0 0 0; 0 0 0; 0 0 0]
	positions = [[i, 0, 0] / n_atoms for i in 0:n_atoms-1]
	atoms = [ElementGaussian(hpot, spread; symbol=:X) for _ in 1:n_atoms]
	unit_cell = (; lattice, positions, atoms)
	# We use a Lennard-Jones alike potential.
	V(x, p) = 4*p.ε * ((p.σ/x)^12 - (p.σ/x)^6)
    params = Dict((:X, :X) => (; ε=1, σ=a / 2^(1/6)))
    PW = PairwisePotential(V, params, max_radius=1e3)
	extra_terms = [PW]
	# If we want, we can also add a non-linear term and a Hartree term.
	do_hartree && (extra_terms = vcat(extra_terms, Hartree()))
	do_nonlinear && (extra_terms = vcat(extra_terms, LocalNonlinearity(ρ -> 2 * ρ^2)))

	(; unit_cell, extra_terms, modes)
end;

# ╔═╡ 98c1a080-b41c-492e-aec6-a4da292677fd
md"""
### The model

We create the model through the `model_1d` function. By default, it will have a kinetic term, a local potential and the Lennard-Jones alike potential.
"""

# ╔═╡ 971eb748-090c-47aa-9033-2c180abb2b63
function model_1d(cell; extra_terms=[], εF, kwargs...)
	n_electrons = isnothing(εF) ? length(cell.positions) : nothing
    @assert !(:terms in keys(kwargs))
    terms = [Kinetic(),
             AtomicLocal(),
             extra_terms...]
    Model(cell.lattice, cell.atoms, cell.positions;
	      model_name="1d system", terms, symmetries=false, n_electrons, εF,
          disable_electrostatics_check=true, spin_polarization=:spinless,
	      kwargs...)
end;

# ╔═╡ ed7200df-7f9b-47b4-a6bd-666676ca883c
md"""
### The convergence parameters

We initialise some reasonable convergence parameters for the system to converge.

All the comparaison we will do here should be exact at machine precision for any set of parameters to the accuracy of the convergence of the density `scf_tol`.
"""

# ╔═╡ 757cec4b-9d48-4001-9f8c-174d292a5c78
function get_convergence_parameters(n_atoms=2; kgrid=(n_atoms, 1, 1))
	Ecut = 20
	fft_size = [25*n_atoms, 1, 1]
	basis_kwargs = (; Ecut, kgrid, fft_size)
	scf_tol = 1e-9
	scf_kwargs = (; tol=scf_tol,
                  determine_diagtol=DFTK.ScfDiagtol(; diagtol_max=scf_tol),
				  callback=identity,
				  eigensolver=diag_full)
	(; basis_kwargs, scf_kwargs)
end;

# ╔═╡ 37a1612c-cd13-4e0d-81f5-7fcbb2db322d
md"""
## Computing the ground state

We have everything we need to compute the ground state of the system. We use the `compute_scfres` as an helper to prepare everything.
"""

# ╔═╡ da30a8bd-cb64-4b34-a4bc-1f60da4aa827
function compute_scfres(n_atoms, εF; kgrid=(n_atoms, 1, 1), kwargs...)
	cell, extra_terms, modes = prepare_unit_cell(n_atoms; kwargs...)
	basis_kwargs, scf_kwargs = get_convergence_parameters(n_atoms; kgrid)

	model = model_1d(cell; extra_terms, εF)
	basis = PlaneWaveBasis(model; basis_kwargs...)

	nbandsalg = isnothing(εF) ? AdaptiveBands(model) : FixedBands(; n_bands_converge=length(cell.positions))
	scfres = self_consistent_field(basis; scf_kwargs..., nbandsalg), modes
end;

# ╔═╡ 7f3b1d64-0e36-4a27-b54e-598f96b6b1a0
md"""
To play with the local potential, we can modify the parameters
* `hpot` $(@bind hpot Scrubbable(range(0.5, stop=10.0, step=0.1), default=3.0)), and
* `spread` $(@bind spread Scrubbable(range(0.1, stop=3.0, step=0.1), default=0.5)),
that will modify  the height and overlap of the potential between atoms.

By default, we choose parameters that have some overlap.


We can also add a Hartree or a non-linear term if we want:
* Hartree term: $(@bind do_hartree CheckBox());
* Non-linear term: $(@bind do_nonlinear CheckBox()).
"""

# ╔═╡ 2fa243cd-6a25-4d5c-b69d-854988f7d0bc
md"""
### SCF computation
"""

# ╔═╡ 2f3efcf1-6eb7-466d-a51e-940ca07eaaf1
begin
	scfres, modes = compute_scfres(n_atoms, εF; hpot, spread, do_hartree, do_nonlinear)
	@info "Occupation numbers" scfres.occupation
	@info "Eigenvalues that are considered" [eig_ik[scfres.occupation[ik] .> scfres.occupation_threshold] for (ik, eig_ik) in enumerate(scfres.eigenvalues)]
end;

# ╔═╡ a4ace2e3-5222-4013-8a3e-02e5c217ed47
md"""
### Plot

Let's see what the density and potential look like.
"""

# ╔═╡ 41e7b0e6-5e51-469e-93b7-08daa57b4fec
let
	x_coords = [x for (x, y, z) in r_vectors_cart(scfres.basis)][:]
	clf()
	local_term = only(filter(e -> typeof(e) <: DFTK.TermAtomicLocal, scfres.basis.terms))
	V = local_term.potential_values[:]
	plot(x_coords, V, label=L"$V$")
	plot(x_coords, scfres.ρ[:], label=L"$ρ$")
	legend()
	gcf()
end

# ╔═╡ 5058b801-4a85-48b9-ad26-8a2e4099aef2
md"""
## Phonon modes

If we want to study the perturbations of nuclei that have the periodicity larger that that of the unit cell, a simple way would be to create a supercell and to compute the Hessian of the supercell system to obtain the so-called _dynamical matrix_.

This can straightforwardly be done in `DFTK` using automatic differentiation on the forces through the `compute_dynmat_ad` function.

This gives us access to all perturbations of the nuclei positions that have the periodicity of a subsystem of the supercell.
"""

# ╔═╡ 17689057-10c4-4b97-8c51-46fe84c7cefe
md"""
### Supercell method

First, let's create a supercell from the unit cell computations. For simplicity, the supercell will be the unfolded version of the cell (i.e., the supercell has only one ``k``-point.)
"""

# ╔═╡ 313c7db0-a7d5-440c-867c-98167771a4d4
dynmat = compute_dynmat_ad(cell_to_supercell(scfres.basis), tol=1e-9);

# ╔═╡ bece2b2d-9885-437d-92ea-017ed4056515
md"""
As we can see from the output, for a one-dimensional problem, we need to do ``n_{\rm atoms}×n_{\rm supercell}`` self-consistent field computations on a ``n_{\rm supercell}`` larger system.

Hence the computational complexity increases significantly, even for a few perturbations.

Non the less, from the dynamical matrix, we can compute the discrete collective modes of vibrations of the system. In the case with a supercell of size `2`, the ones that have the periodicity of the unit cell and twice that.
"""

# ╔═╡ 77a753ff-f4bc-4f81-981e-f556f8c7b31a
phonons = let
	λs, Vs = eigen(dynmat)
	λs[abs.(λs) .< 1e-6] .= 0 
	[Dict(:λ => sqrt(λs[i]), :d => Vs[:, i]) for i in eachindex(λs)]
end;

# ╔═╡ ce55bc11-57d1-4b43-82da-a31e368945eb
md"""
To each eigenvalue ``λ`` in Hartree, is associated a displacement of the nuclei ``d``.

For example for a system with ``2`` atoms and a supercell of ``4`` atoms, the lowest eigenvalue ``0`` corresponds to the movements of all the atoms in the same direction 
* $(join(round.(phonons[1][:d]; digits=2), ", ") |> Text)
and the highest one to movements of the atoms in the first unit cell in one direction and the other one in the opposite
* $(join(round.(phonons[end][:d]; digits=2), ", ") |> Text)

We note that the first eigenvalue could have been obtained with computations on the unit cell, but not the second.
"""

# ╔═╡ dcb7e8ed-74f6-48f4-9413-1cfb5838eef0
md"""
### Unit cell computations

To transform a problem that _needs_ information larger than the supercell, we need to transform the problem by using perturbation theory.

This information is already present in materials computations through the use of ``k``-points.

Indeed, while the potential and density are unit cell periodic, the Hamiltonian of the system is block-diagonalised by factorising the wave functions into functions
```math
ψ_{kn}(x) = e^{ik·x} u_{nk}(x),
```
where ``u_{nk}`` is unit cell periodic.
"""

# ╔═╡ 97d9b9f1-fe6c-434a-9752-41fbf5399a16
md"""
### Expliquer

Taylor, harmonic approximation, …

```math
\begin{aligned}
δu &= e^{iq·x} δx \\
δV &= e^{iq·x} δV^q \\
(H_0^k - ε_{n,k-q}) [δψ_{n,k-q}]_k &= -P_k^\perp [[δV^q] u_{n,k-q}] \\
δρ^q &= χ_0^q δV^q \\
δρ &= e^{iq·x} δρ^q
\end{aligned}
```
"""

# ╔═╡ cee21c8d-251b-4184-a449-f064a96d1b33
md"""
### Comparison between perturbation and unit cell computations

#### Preparatory work

We now compare the supercell results with results obtained from perturbation theory. For this, we need to extract from the results at the ground state the quantities we want to compare. This is done with `get_quantities_of_interest`.
"""

# ╔═╡ 4cd360f1-8ced-404f-8f81-596aa84f9e48
function get_quantities_of_interest(scfres)
	# The x coordinates of the system.
	x_coords = [x for (x, y, z) in r_vectors_cart(scfres.basis)][:]

	# The potential, the ground state wave functions and density.
	local_term = only(filter(e -> typeof(e) <: DFTK.TermAtomicLocal, scfres.basis.terms))
	V = local_term.potential_values[:]
	ψ = [ψk[:] for ψk in scfres.ψ]
	ρ = scfres.ρ[:]

	# Extract the occupied states.
	mask_occ = map(occk -> findall(isless.(scfres.occupation_threshold, occk)), scfres.occupation)

	# And the perturbations of different quatities with repspect to independent displacements of the atoms.
	# The displacements we can study are the ones of the form ``e^{iq·x}``, where ``q`` is in the ``k``-points.
	qpoints = getfield.(scfres.basis.kpoints, :coordinate)
	δVs = [DFTK.compute_δV(local_term, scfres.basis; q) for q in qpoints]

	# The quantities ``δV_q ψ_{k-q}``,
	δVψs = [DFTK.compute_δHψ(scfres; q) for q in qpoints]
	# the perturbations δψ,
	δψs = [DFTK.compute_δρ(scfres; q).δψs for q in qpoints]
	# and finally the perturbations of the density.
	δρs = [DFTK.compute_δρ(scfres; q).δρs for q in qpoints]

	(; x_coords, V, ψ, ρ, qpoints, δVs, δVψs, δψs, δρs, scfres, mask_occ)
end;

# ╔═╡ a32990c9-7f5c-48ec-b2c4-ad7c65e0d964
md"""
We compute them for the unit cell computation…
"""

# ╔═╡ b7e3129e-4c7e-4b18-aa4a-e35cb8de8cbd
cell_qoe = get_quantities_of_interest(scfres);

# ╔═╡ ab2a6c4c-36a6-4fbd-8df4-cdb232b3e8b3
md"""
… as well as for the supercell computations.
"""

# ╔═╡ a094da30-42cd-4d9d-bb3d-d8a27c87cb81
supercell_qoe, supercell_size = let
	supercell_scfres = cell_to_supercell(scfres)
	supercell_size = scfres.basis.kgrid
	supercell_qoe = get_quantities_of_interest(supercell_scfres)
	(; supercell_qoe, supercell_size)
end;

# ╔═╡ d6d87cb2-ae0d-4039-8e76-a200620c7802
md"""
!!! warning "Shortcut"
	By unfolding the results from the cell to the supercell, we have taken a shortcut. To independently compare the two, we should create another system with four atoms.

	```
	supercell_scfres, _ = compute_scfres(n_atoms*prod(supercell_size), εF; hpot, spread, do_hartree, do_nonlinear, kgrid=(1, 1, 1))
	```
"""

# ╔═╡ 39f57c52-8674-41f8-9a99-a4fae1ecc9ee
md"""
Remember the `+` or `-` modes we briefly mentioned before? They are needed to know which sign to put in the linear combinations of elements to be able to map the results from the unit cell to the results of the supercell. The function below helps us to do that.
"""

# ╔═╡ 77a06f6d-8aa2-46bd-a5b6-a48af62d81cd
begin
	function cell_to_supercell_modes(modes, supercell_scfres, supercell_size, qpoint)
		@assert all(m ∈ [:+, :-] for m in modes)
		modes_real = [m == :+ ? 1.0 : -1.0 for m in modes]
	
		modes_supercell = map(modes_real) do m
			m_supercell = []
			for Rx in 0:supercell_size[1]-1
				sign = DFTK.cis2pi(dot(qpoint, [Rx, 0, 0]))
				@assert iszero(norm(imag(sign)))
				sign = real(sign)
				push!(m_supercell, sign * m)
			end
			m_supercell
		end
		factors_supercell = reshape(hcat(modes_supercell...)', (1, :))[:]
		# Reorder with respect to the positions of the supercell atoms.
		factors_supercell = factors_supercell[sortperm(supercell_scfres.basis.model.positions)]
		(; cell=modes_real, supercell=factors_supercell)
	end

	factors = [cell_to_supercell_modes(modes, supercell_qoe.scfres, supercell_size, q) for q in cell_qoe.qpoints]
end;

# ╔═╡ 2f1b7c3a-8239-4544-acdd-ff9e13bf78de
md"""
We also introduce two helper functions. The first one to have two make identic copies of some quantities, the second one so we do not have to care about renormalisation and constant phase factor changes between results, and the third one is the ``e^{iq·x}`` function on the supercell, where ``q`` is the ``q``-point.
"""

# ╔═╡ 9df324fd-400d-4fa6-be10-5420d37950c2
begin
	vcat2(x) = vcat(x, x)
	normalize(x) = (x .* conj(x[1])) / norm(x .* conj(x[1]))
	exp_iqx = [DFTK.cis2pi.(-dot.(Ref([Rx, 0, 0]), r_vectors(supercell_qoe.scfres.basis)))[:] for Rx in 0:supercell_size[1]-1]
end;

# ╔═╡ 1b9ef7e0-6333-4e02-bfb9-51ae6e987830
md"""
Finally, we will modify a bit the raw results to easily be able to plot the quantities. We map everything to plot in the supercell.
"""

# ╔═╡ 07cdbbf2-2441-4ee2-b6d1-800ec74a58b9
function cell_to_supercell_qoe(qoe, supercell_size)
	# Number of copies to do.
	n = prod(supercell_size)

	scfes = qoe.scfres
	ψ = scfres.ψ
	basis = scfres.basis

	x_coords = qoe.x_coords
	for _ in 1:n-1
		x_coords = vcat(x_coords, x_coords[end] + x_coords[2] .+ x_coords)
	end
	@assert length(x_coords) == prod(supercell_size) * length(qoe.x_coords)

	ρ = qoe.ρ
	for _ in 1:n-1
		ρ = vcat2(ρ)
	end

	V = qoe.V
	for _ in 1:n-1
		V = vcat2(V)
	end

	ψ_real = [Array{eltype(ψ[ik])}(undef, (length(x_coords), size(ψ[ik], 2))) for ik in 1:length(ψ)]
	for (ik, kpt) in enumerate(basis.kpoints)
		for n in 1:size(ψ[ik], 2)
			ψ_real[ik][:, n] = normalize(vcat2(ifft(basis, kpt, ψ[ik][:, n])))
		end
	end

	local_term = only(filter(e -> typeof(e) <: DFTK.TermAtomicLocal, basis.terms))
	δVs = [DFTK.compute_δV(local_term, basis; q) for q in qoe.qpoints]
	δVs = [[vcat2(δV[:]) for δV in δVq] for δVq in δVs]

	δρs = [DFTK.compute_δρ(scfres; q).δρs for q in qoe.qpoints]
	δρs = [[vcat2(δρ[:]) for δρ in δρq] for δρq in δρs]
	
	(; x_coords, V, ρ, ψ=ψ_real, δVs, δρs)
end;

# ╔═╡ 8baa886b-6652-4008-b628-ecb7eb4b60c7
function supercell_to_supercell_qoe(qoe)
	x_coords = qoe.x_coords
	ρ = qoe.ρ
	V = qoe.V

	scfres = qoe.scfres
	ψ = scfres.ψ
	basis = scfres.basis

	ψ_real = [Array{eltype(ψ[ik])}(undef, (length(x_coords), size(ψ[ik], 2))) for ik in 1:length(ψ)]
	for (ik, kpt) in enumerate(basis.kpoints)
		for n in 1:size(ψ[ik], 2)
			ψ_real[ik][:, n] = normalize(ifft(basis, kpt, ψ[ik][:, n]))
		end
	end

	local_term = only(filter(e -> typeof(e) <: DFTK.TermAtomicLocal, basis.terms))
	δVs = [δV[:] for δV in DFTK.compute_δV(local_term, basis)]

	δρs = [δρ[:] for δρ in DFTK.compute_δρ(scfres).δρs]

	(; x_coords, ρ, V, ψ=ψ_real, δVs, δρs)
end;

# ╔═╡ c0192298-72e1-49d0-a5dd-00d90fdff393
begin
	tcell_qoe = cell_to_supercell_qoe(cell_qoe, supercell_size)
	tsupercell_qoe = supercell_to_supercell_qoe(supercell_qoe)
end;

# ╔═╡ 90d592e4-e76a-41ec-b5ae-9efde9572e83
md"""
#### Plot

Okay, we are finally done. We can now plot the results of the different computations and observe that they are indeed equal to machine accuracy.
"""

# ╔═╡ 51c30baf-4136-47ff-91f4-73059cd52066
md"""
##### ``ψ``
"""

# ╔═╡ aae12a40-ad9c-496b-95e5-26c486674c74
begin
	n_plots = length(collect(Iterators.flatten(cell_qoe.mask_occ)))
	clf()
	figures_ψ = []
	for ik in 1:length(tcell_qoe.ψ)
		for n in 1:length(cell_qoe.mask_occ[ik])
			push!(figures_ψ, figure())
			idx = n + (ik-1)*(length(cell_qoe.mask_occ[ik]))

			suptitle(L"$ψ_{"*string(idx)*L"}$")

			mask = supercell_qoe.mask_occ[1][idx]

			ψ_cell = tcell_qoe.ψ[ik][:, n] .* exp_iqx[ik]

			ψ_supercell = tsupercell_qoe.ψ[1][:, mask]

			subplot(321)	
			plot(real(ψ_cell))
			title("Cell, real")
	
			subplot(322)	
			plot(imag(ψ_cell))
			title("Cell, imag")
	
			subplot(323)
			plot(real(ψ_supercell))
			title("Supercell, real")

			subplot(324)
			plot(imag(ψ_supercell))
			title("Supercell, imag")

			subplot(325)
			plot(real(ψ_cell - ψ_supercell))
			title("Error, real")
			subplot(326)
			plot(imag(ψ_cell - ψ_supercell))
			title("Error, imag")
		end
	end
	figures_ψ
end;

# ╔═╡ 29b4e3e8-932f-45d9-aceb-b22b824f46c4
@bind iψ Slider(1:length(figures_ψ), default = 1)

# ╔═╡ 4e959952-f314-47cd-8ded-f8d4213e3d45
figures_ψ[iψ]

# ╔═╡ a85fd23d-2b41-47e0-ba03-c4bb1fb16b10
md"""
##### ``δV``
"""

# ╔═╡ c2d9ad24-84c5-4673-a227-2cccbc9aa95b
begin
	clf()
	figures_δV = []
	for ik in 1:length(tcell_qoe.δVs)
		δV_cell = sum(factors[ik].cell[n] .* tcell_qoe.δVs[ik][n] for n in 1:size(tcell_qoe.δVs[ik], 2)) .* exp_iqx[ik]

		δV_supercell = sum(factors[ik].supercell[n] .* tsupercell_qoe.δVs[n] for n in 1:length(factors[ik].supercell)) / prod(supercell_size)

		push!(figures_δV, figure())
		suptitle(L"$δV_{"*string(cell_qoe.qpoints[ik][1])*L"}$")

		res_cell = δV_cell
		res_supercell = δV_supercell

		subplot(321)	
		plot(real(res_cell))
		title("Cell, real")

		subplot(322)	
		plot(imag(res_cell))
		title("Cell, imag")

		subplot(323)
		plot(real(res_supercell))
		title("Supercell, real")

		subplot(324)
		plot(imag(res_supercell))
		title("Supercell, imag")

		subplot(325)
		plot(real(res_cell - res_supercell))
		title("Error, real")
		subplot(326)
		plot(imag(res_cell - res_supercell))
		title("Error, imag")
	end
	figures_δV
end;

# ╔═╡ ecb42815-7acd-4187-a90b-a69c74d9f60e
@bind iδV Slider(1:length(figures_δV), default = 1)

# ╔═╡ 690d54d5-83e6-4e95-8626-c19c3cdce214
figures_δV[iδV]

# ╔═╡ bbd3c614-9bd5-4a70-884a-ebc337483e16
md"""
##### ``δρ``
"""

# ╔═╡ 7bd88814-b017-4b6b-9e0c-da90ab2ee1c8
begin
	clf()
	figures_δρ = []
	for ik in 1:length(tcell_qoe.δρs)
		δρ_cell = sum(factors[ik].cell[n] .* tcell_qoe.δρs[ik][n] for n in 1:size(tcell_qoe.δρs[ik], 2)) .* exp_iqx[ik]

		δρ_supercell = sum(factors[ik].supercell[n] .* tsupercell_qoe.δρs[n] for n in 1:length(factors[ik].supercell)) / prod(supercell_size)

		push!(figures_δρ, figure())
		suptitle(L"$δρ_{"*string(cell_qoe.qpoints[ik][1])*L"}$")

		res_cell = δρ_cell
		res_supercell = δρ_supercell

		subplot(321)	
		plot(real(res_cell))
		title("Cell, real")

		subplot(322)	
		plot(imag(res_cell))
		title("Cell, imag")

		subplot(323)
		plot(real(res_supercell))
		title("Supercell, real")

		subplot(324)
		plot(imag(res_supercell))
		title("Supercell, imag")

		subplot(325)
		plot(real(res_cell - res_supercell))
		title("Error, real")
		subplot(326)
		plot(imag(res_cell - res_supercell))
		title("Error, imag")
	end
	figures_δρ
end;

# ╔═╡ 0b19dd51-563b-4168-8a65-4423d1fe0cc3
@bind iδρ Slider(1:length(figures_δρ), default = 1)

# ╔═╡ ae502d85-01a6-4b8a-b2a8-b1806df9dcc2
figures_δρ[iδρ]

# ╔═╡ 741e8787-0f24-463f-91c1-cecb6b765002
md"""
##### Other quantities

We could also plot the right-hand side ``δV^qψ_{n,k-q}``, the projection of the right-hand side on occupied space, and the ``δψ``.
"""

# ╔═╡ 95b6d1cc-78f0-481b-8f93-ec6d1694a8d4
md"""
# TODO

Plot the modes
"""

# ╔═╡ 9b9b6789-ad05-410a-b18c-5db7732c3288
# We have now every thing we need to compare the results of the cell and supercell.
function myplot(terms, x_coords, qoe, factors; scale=identity, qpoint=zeros(3), celltype=:cell)
	function merge_cplots(cplots, part, p)
		if part == real
			merge(cplots, (; real=p))
		else
			merge(cplots, (; imag=p))
		end
	end

	qpoints_colors = [:blue, :orange]
	plots = (; )
	part = :imag ∈ terms ? imag : real
	for term in terms
		if term == :modes
			cplots = (; )
			idx_qpt = only(findall(q -> q == qpoint, qoe.qpoints))
			cfactors = factors[idx_qpt][celltype]
			markers = [factor > 0 ? 6 : 7 for factor in cfactors]
			ticks_length = length(x_coords) / length(cfactors) / 100
			ticks = [ticks_length*i*x_coords[end] for i in 0:length(cfactors)-1]
			for (i, tick) in enumerate(ticks)
				p = plot([tick], [0]; marker=markers[i], color=qpoints_colors[idx_qpt])
				cplots = merge_cplots(cplots, part, p)
			end
			plots = merge(plots, (; V=cplots))
		end
		if term == :V
			cplots = (; )
			p = plot(x_coords, part(scale(qoe.V)), label=L"$V$")
			cplots = merge_cplots(cplots, part, p)
			plots = merge(plots, (; V=cplots))
		end
		if term == :ρ
			cplots = (; )
			p = plot(x_coords, part(scale(qoe.ρ)), label=L"$ρ$")
		 	cplots = merge_cplots(cplots, part, p)
			plots = merge(plots, (; ρ=cplots))
		end
	end
	plots
end;

# ╔═╡ 73cabca3-5eb8-484c-ad30-0d3f42fd0cd5
begin
	clf()
	suptitle("Supercell")

	subplot(221)
	myplot([:V, :ρ, :real, :modes], supercell_qoe.x_coords, supercell_qoe, factors, celltype=:supercell)
	legend()
	title("Real part")

	subplot(222)
	myplot([:V, :ρ, :imag, :modes], supercell_qoe.x_coords, supercell_qoe, factors, celltype=:supercell)
	legend()
	title("Imaginary part")

	subplot(223)
	myplot([:modes], supercell_qoe.x_coords, cell_qoe, factors, celltype=:cell, qpoint=cell_qoe.qpoints[1])
	myplot([:modes], supercell_qoe.x_coords, cell_qoe, factors, celltype=:cell, qpoint=cell_qoe.qpoints[2])
	gcf()
end

# ╔═╡ 0c3db408-ce32-40e0-bcf4-3c23da0b1d90
begin
	clf()
	suptitle("Unit cell")

	subplot(121)
	myplot([:V, :ρ, :real, :modes], supercell_qoe.x_coords, cell_qoe, factors; scale=vcat2, celltype=:supercell)
	legend()
	title("Real part")

	subplot(122)
	myplot([:V, :ρ, :imag, :modes], supercell_qoe.x_coords, cell_qoe, factors; scale=vcat2, celltype=:supercell)
	legend()
	title("Imaginary part")
	gcf()
end

# ╔═╡ Cell order:
# ╟─2d4bea83-e31c-424f-b73e-30c0b7e97ce7
# ╟─a24e6313-0e18-4e05-99ce-8ca6133fde33
# ╠═1d1035e2-a2d9-4879-ae02-01775731fb61
# ╠═b53cf69c-3cf8-44bf-8fef-5483ab1b382d
# ╠═b7594e68-a789-433f-acc8-c0bc9ba432a2
# ╟─e4b5ebfa-ff93-4945-888a-fa8e69d80aa8
# ╠═01debbd5-357a-4494-8586-3ff98883dd34
# ╟─bbae56f8-0e3d-4497-97cd-704f5d2c6124
# ╟─f741a53c-7bb6-4f80-bdf6-551aeef1fec7
# ╟─6dc3d7de-c435-4092-8057-5721188a71db
# ╠═8b3215b4-8e02-40dc-83d3-95521c21fbdc
# ╠═98c1a080-b41c-492e-aec6-a4da292677fd
# ╠═971eb748-090c-47aa-9033-2c180abb2b63
# ╟─ed7200df-7f9b-47b4-a6bd-666676ca883c
# ╠═757cec4b-9d48-4001-9f8c-174d292a5c78
# ╟─37a1612c-cd13-4e0d-81f5-7fcbb2db322d
# ╠═da30a8bd-cb64-4b34-a4bc-1f60da4aa827
# ╟─7f3b1d64-0e36-4a27-b54e-598f96b6b1a0
# ╟─2fa243cd-6a25-4d5c-b69d-854988f7d0bc
# ╠═2f3efcf1-6eb7-466d-a51e-940ca07eaaf1
# ╟─a4ace2e3-5222-4013-8a3e-02e5c217ed47
# ╠═41e7b0e6-5e51-469e-93b7-08daa57b4fec
# ╟─5058b801-4a85-48b9-ad26-8a2e4099aef2
# ╟─17689057-10c4-4b97-8c51-46fe84c7cefe
# ╠═313c7db0-a7d5-440c-867c-98167771a4d4
# ╟─bece2b2d-9885-437d-92ea-017ed4056515
# ╠═77a753ff-f4bc-4f81-981e-f556f8c7b31a
# ╟─ce55bc11-57d1-4b43-82da-a31e368945eb
# ╟─dcb7e8ed-74f6-48f4-9413-1cfb5838eef0
# ╟─97d9b9f1-fe6c-434a-9752-41fbf5399a16
# ╟─cee21c8d-251b-4184-a449-f064a96d1b33
# ╠═4cd360f1-8ced-404f-8f81-596aa84f9e48
# ╟─a32990c9-7f5c-48ec-b2c4-ad7c65e0d964
# ╠═b7e3129e-4c7e-4b18-aa4a-e35cb8de8cbd
# ╠═ab2a6c4c-36a6-4fbd-8df4-cdb232b3e8b3
# ╠═a094da30-42cd-4d9d-bb3d-d8a27c87cb81
# ╟─d6d87cb2-ae0d-4039-8e76-a200620c7802
# ╟─39f57c52-8674-41f8-9a99-a4fae1ecc9ee
# ╠═77a06f6d-8aa2-46bd-a5b6-a48af62d81cd
# ╟─2f1b7c3a-8239-4544-acdd-ff9e13bf78de
# ╠═9df324fd-400d-4fa6-be10-5420d37950c2
# ╟─1b9ef7e0-6333-4e02-bfb9-51ae6e987830
# ╠═07cdbbf2-2441-4ee2-b6d1-800ec74a58b9
# ╠═8baa886b-6652-4008-b628-ecb7eb4b60c7
# ╠═c0192298-72e1-49d0-a5dd-00d90fdff393
# ╟─90d592e4-e76a-41ec-b5ae-9efde9572e83
# ╟─51c30baf-4136-47ff-91f4-73059cd52066
# ╠═aae12a40-ad9c-496b-95e5-26c486674c74
# ╠═29b4e3e8-932f-45d9-aceb-b22b824f46c4
# ╠═4e959952-f314-47cd-8ded-f8d4213e3d45
# ╟─a85fd23d-2b41-47e0-ba03-c4bb1fb16b10
# ╠═c2d9ad24-84c5-4673-a227-2cccbc9aa95b
# ╠═ecb42815-7acd-4187-a90b-a69c74d9f60e
# ╠═690d54d5-83e6-4e95-8626-c19c3cdce214
# ╟─bbd3c614-9bd5-4a70-884a-ebc337483e16
# ╠═7bd88814-b017-4b6b-9e0c-da90ab2ee1c8
# ╠═0b19dd51-563b-4168-8a65-4423d1fe0cc3
# ╠═ae502d85-01a6-4b8a-b2a8-b1806df9dcc2
# ╟─741e8787-0f24-463f-91c1-cecb6b765002
# ╟─95b6d1cc-78f0-481b-8f93-ec6d1694a8d4
# ╠═9b9b6789-ad05-410a-b18c-5db7732c3288
# ╠═73cabca3-5eb8-484c-ad30-0d3f42fd0cd5
# ╠═0c3db408-ce32-40e0-bcf4-3c23da0b1d90
