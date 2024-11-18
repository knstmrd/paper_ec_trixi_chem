using Trixi
include("../src/TrixiChem.jl")
using .TrixiChem
using OrdinaryDiffEq
using Trixi2Vtk

@inline function source_terms_Arrhenius_diss_only_scaling(u, x, t,
                                                  equations::CompressibleEulerEquationsMs1T2D)

    T = TrixiChem.temperature(u, equations) * T_ref
    n_mol = u[4] / equations.mass[1] # scaled
    n_atom = u[5] / equations.mass[2] # scaled

    exp_E_diss =  exp(-E_diss / T)
    k_diss_m_a = A_m_a * T^n_m_a * exp_E_diss # 0.0  # non-scaled
    k_diss_m_m = A_m_m * T^n_m_m * exp_E_diss # 0.0  # non-scaled
 
    dn_mol_diss = -k_diss_m_a * n_mol * n_atom * k_scaler
    dn_mol_diss -= k_diss_m_m * n_mol * n_mol * k_scaler

    dn_atom_diss = -2 * dn_mol_diss

    drho_e = 0.0
    # velocity remains unchanged 

    drho_mol = dn_mol_diss * equations.mass[1]
    drho_atom = dn_atom_diss * equations.mass[2]
    return SVector(0.0, 0.0, drho_e, drho_mol, drho_atom)
end



@inline function initial_condition(x, t, equations::CompressibleEulerEquationsMs1T2D)
    prim = SVector(v1_freestream / v_ref, v2_freestream / v_ref, T_freestream / T_ref, rho_mol / rho_ref, rho_atom / rho_ref)
    return prim2cons(prim, equations)
end


function cons2prim_scaled(u, equations::CompressibleEulerEquationsMs1T2D)
    rho_v1, rho_v2, rho_e, rho_mol, rho_atom = u

    rho = rho_mol + rho_atom

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho

    T = TrixiChem.temperature(u, equations)
    p = pressure(u, equations)

    return SVector(v1 * v_ref, v2 * v_ref, p * p_ref, T * T_ref, rho_mol * rho_ref, rho_atom * rho_ref)
end

Trixi.varnames(::typeof(cons2prim_scaled), ::CompressibleEulerEquationsMs1T2D) = ("v1", "v2", "p", "T", "rho_mol", "rho_atom")

global const Θvibr = 2273.54  # O2
global const E_diss = 59364.8  # O2
global const E_form_O2 = 0.0
global const E_form_O = 0.5 * E_diss

global const mass_mol = 5.3134e-26  # O2
global const mass_atom = mass_mol / 2  # O

global const v1_freestream = 0.0
global const v2_freestream = 0.0
global const T_freestream = 12000.0
global const p_freestream = 1e23 * k_B * T_freestream

global const x_mol = 0.5
global const x_atom = 1.0 - x_mol
global const n = p_freestream / (k_B * T_freestream)
global const rho_mol = x_mol * n * mass_mol
global const rho_atom = x_atom * n * mass_atom

global const tmp_e_v_arr = generate_e_vibr_arr_harmonic_K(Θvibr, E_diss)

global const e_int_mol = T -> e_rot_cont(mass_mol, T) + e_vibr_from_array(mass_mol, T, tmp_e_v_arr) + E_form_O2 * k_B / mass_mol
global const c_int_mol = T -> c_rot_cont(mass_mol, T) + c_vibr_from_array(mass_mol, T, tmp_e_v_arr)

global const e_int_atom = T -> 0.0 * T + E_form_O * k_B / mass_atom
global const c_int_atom = T -> 0.0 * T

global const A_m_a = 1.6605e-8
global const n_m_a = -1.5

global const A_m_m = 3.321e-9
global const n_m_m = -1.5

global const L_ref = 1.0
global const T_ref = T_freestream
global const p_ref = p_freestream
global const m_ref = mass_mol
global const e_ref = k_B * T_ref / mass_mol
global const cv_ref = k_B / mass_mol
global const n_ref = p_ref / (T_ref * k_B)
global const rho_ref = n_ref * mass_mol
global const v_ref = sqrt(p_ref / rho_ref)
global const t_ref = L_ref / v_ref
global const t_max = 5000 * 1e-7
global const t_max_scaled = t_max / t_ref
global const ntsteps = 5000
global const dt_output = t_max_scaled / ntsteps
global const k_scaler = m_ref * n_ref^2 * t_ref / (rho_ref)  # scaling for reaction rate coefficients

println("t_ref: $t_ref, t_max_scaled: $t_max_scaled")

global const polydeg = 2

equations = CompressibleEulerEquationsMs1T2D([mass_mol, mass_atom], [e_int_mol, e_int_atom], [c_int_mol, c_int_atom],
                                             mass_ref=m_ref, T_ref=T_ref, e_ref=e_ref, T_tol=1e-11, min_T_jump_rel=1e-6)


outpref = "output/O2_12k/" * string(polydeg)
outdir = outpref * "/hdf"
outdirvtk = outpref * "/vtk"

mkpath(outdir)
mkpath(outdirvtk)

volume_flux  = flux_oblapenko
surface_flux = flux_oblapenko

solver = DGSEM(polydeg=3, surface_flux=surface_flux,
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-1.0, -1.0)
coordinates_max = ( 1.0,  1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=1,
                n_cells_max=10_000,
                periodicity=true)

                
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_condition_periodic, source_terms=source_terms_Arrhenius_diss_only_scaling)



save_solution = SaveSolutionCallback(dt=dt_output,
                                     #interval=250,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim_scaled,
                                     output_directory=outdir)

analysis_callback = AnalysisCallback(semi, interval=10)

callbacks = CallbackSet(analysis_callback,
                        #alive_callback,
                        #stepsize_callback,
                        save_solution)

# ODE solvers
tspan = (0.0, t_max_scaled)
ode = semidiscretize(semi, tspan);
sol = solve(ode, SSPRK43(); abstol=1.0e-6, reltol=1.0e-6,
            ode_default_options()..., callback=callbacks)

trixi2vtk(joinpath(outdir, "solution_*.h5"), output_directory=outdirvtk)
