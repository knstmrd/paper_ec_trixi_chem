# Multi-species reacting flow simulations with Trixi
This repository contains code for the multi-species chemically non-equilibrium reacting Euler equations with *arbitrary* internal energy functions
of the constituent species, based on [Trixi.jl](https://github.com/trixi-framework/Trixi.jl). This is an extension of the
single-species case (see [Oblapenko, Torrilhon 2024](https://arxiv.org/abs/2403.16882) for the derivation of the fluxes in the single-species case
as well as the [associated reproducibility repo](https://github.com/knstmrd/paper-ec_trixi_inte)).

Thermal equilibrium is assumed (i.e. all the internal modes of all species have a single temperature T).

The code might run quite slow, as multiple quantities such as temperature are computed multiple times for the same timestep; other things are not super-optimized as well.

## Usage
To use the code, clone the git project, run `julia --project=.` from the root dir and then run `using Pkg; Pkg.resolve(); Pkg.instantiate();`.

One can run examples from the `examples` dir by running `julia --project=. examples/example_name.jl` (`example_name.jl` being a placeholder for the specific file).
By default the output is written to the `output` directory in the project, this can be changed by changing the `outpref` variable in the example files.

## CompressibleEulerEquationsMs1T2D
The `CompressibleEulerEquationsMs1T2D` type implements the multi-species 2-D Euler equations for gases with arbitrary internal energies.
First one needs to set up the functions for the specific energy functions (units of J/kg) and specific heats (units of J/K/kg).

One can use the various functions provided in `energy_utils.jl` to construct these internal energy-related functions (the translational energy component of 3/2kT
is accounted for automatically!).
For example, for an O2/O mixture:
```julia

Θvibr = 2273.54  # characteristic vibrational temperature of O2, K
E_diss = 59364.8  # dissociation energy of O2, K
E_form_O2 = 0.0  # specific enthalpy of formation of O2, K 
E_form_O = 0.5 * E_diss  # specific enthalpy of formation of O, K

mass_mol = 5.3134e-26  # mass of O2, kg
mass_atom = mass_mol / 2  # mass of O, kg

tmp_e_v_arr = generate_e_vibr_arr_harmonic_K(Θvibr, E_diss)  # generate cut-off harmonic oscillator spectrum for O2

# e(T), cv(T) for O2
e_int_mol = T -> e_rot_cont(mass_mol, T) + e_vibr_from_array(mass_mol, T, tmp_e_v_arr) + E_form_O2 * k_B / mass_mol
c_int_mol = T -> c_rot_cont(mass_mol, T) + c_vibr_from_array(mass_mol, T, tmp_e_v_arr)

# e(T), cv(T) for O2
e_int_atom = T -> 0.0 * T + E_form_O * k_B / mass_atom
c_int_atom = T -> 0.0 * T

m_ref = mass_mol  # reference mass
T_ref = 1000.0  # reference temperature
e_ref = k_B * T_ref / mass_mol  # reference specific energy
```

Then one can instantiate the 2-D multi-species compressible Euler equations:
```julia
equations = CompressibleEulerEquationsMs1T2D([mass_mol, mass_atom], [e_int_mol, e_int_atom], [c_int_mol, c_int_atom],
                                             mass_ref=m_ref, T_ref=T_ref, e_ref=e_ref, T_tol=1e-11, min_T_jump_rel=1e-6,
                                             T_min=10.0, T_max=3.0e4, ΔT=1.0)
```
Here one needs to pass in a list of molecular masses of the species (units of kg), specific energy functions (units of J/kg), specific heats (units of J/K/kg),
a reference mass (units of kg), temperature (units of K), energy (units of J), a tolerance for the Newton solver for temperature, a parameter governing the switch
from evaluation of jump in energy to a midpoint evaluation of specific heat (for numerical stability; more details in the [single-species preprint](https://arxiv.org/abs/2403.16882)),
the minimum and maximum temperatures for the tabulation (in K), and a temperature step size for the tabulation (in K).

**Note** The user is to take care of correct and consistent scaling themselves!

One can then use these equations as desired, with the `flux_oblapenko` function providing an entropy-conservative flux function
(implemented only for Cartesian meshes; `FluxRotated(flux_oblapenko)` needs to be used otherwise).

## Included examples

### Weak blast wave
The `iho_blastwave_2species.jl` example models a weak blast wave using the entropy-conservative fluxes (no dissipation, no limiting).
The vibrational spectrum of O2 is modelled with an infinite harmonic oscillator.

### Spatially homogeneous chemical relaxation
The `chem_2species.jl` example computes the spatially homogeneous relaxation of an O2/O mixture with dissociation reactions (defined in `source_terms_Arrhenius_diss_only`).
The vibrational spectrum of O2 is modelled with a cut-off harmonic oscillator.

### Mach 10 flow around a cylinder
The `cylinder_2species.jl` example computes a Mach 10 flow of an O2/O mixture with dissociation around a cylinder.
It uses a [P4Est grid](https://p4est.org/) with AMR based on a [shock indicator](https://doi.org/10.1016/j.jcp.2020.109935) and the [IDP limiter of Rueda-Ramirez et al.](https://doi.org/10.1016/j.compfluid.2022.105627).
Two command-line parameters need to be supplied: the polynomial degree and the CFL number (CFL of 0.4 seems to work well with up to polynomial orders of 4, higher CFL values may lead to crashes).
The simulation is started on a uniform curved 30x30 grid, the mapping of a square onto this grid is defined by the `mapping_full` function.
The function takes 3 points that defined the the curve of the inflow boundary: `[p1, p2, p3]` is the input list, and the corresponding
points are taken to be `[(-p1, 0), (-p2, p2), (p3, 0)]`.

## Citing
For now, the [single-species preprint](https://arxiv.org/abs/2403.16882) can be cited, the multi-species preprint will be made available later:

```bibtex
@article{oblapenko2024entropy,
  title={Entropy-conservative high-order methods for high-enthalpy gas flows},
  author={Oblapenko, Georgii and Torrilhon, Manuel},
  journal={arXiv preprint arXiv:2403.16882},
  year={2024}
}
```