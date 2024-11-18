using OrdinaryDiffEq
using Trixi
include("../src/TrixiChem.jl")
using .TrixiChem
using Trixi2Vtk

function initial_condition_weak_blast_wave(x, t,
                                           equations::CompressibleEulerEquationsMs1T2D)
    # From Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
    # Set up polar coordinates
    inicenter = SVector(0.0, 0.0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)
    phi = atan(y_norm, x_norm)
    sin_phi, cos_phi = sincos(phi)

    # Calculate primitive variables
    rho = r > 0.5 ? 1.0 : 1.1691
    v1 = r > 0.5 ? 0.0 : 0.1882 * cos_phi
    v2 = r > 0.5 ? 0.0 : 0.1882 * sin_phi
    
    T = r > 0.5 ? 1.0 : 1.06492

    x_mol = 0.5
    n_tot = rho / (0.5 * (x_mol + 1.0))

    rho1 = n_tot * x_mol
    rho2 = 0.5 * n_tot * (1.0 - x_mol)

    return prim2cons(SVector(v1, v2, T, rho1, rho2), equations)
end



function cons2prim_scaled(u, equations::CompressibleEulerEquationsMs1T2D)
    rho_v1, rho_v2, rho_e, rho_mol, rho_atom = u

    rho = rho_mol + rho_atom

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho

    T = temperature(u, equations)
    p = pressure(u, equations)

    return SVector(v1 * v_ref, v2 * v_ref, p * p_ref, T * T_ref, rho_mol * rho_ref, rho_atom * rho_ref)
end


Trixi.varnames(::typeof(cons2prim_scaled), ::CompressibleEulerEquationsMs1T2D) = ("v1", "v2", "p", "T", "rho_mol", "rho_atom")
global const k_B = 1.380649e-23
global const Θvibr = 3393.48
global const T_ref = 1000.0
global const E_diss = 59364.8  # O2
global const mass_mol = 5.3134e-26  # O2
global const mass_atom = mass_mol / 2  # O
global const m_ref = mass_mol
global const e_ref = k_B * T_ref / m_ref
global const p_ref = 101325.0
global const e_ref = k_B * T_ref / mass_mol
global const cv_ref = k_B / mass_mol
global const n_ref = p_ref / (T_ref * k_B)
global const rho_ref = n_ref * mass_mol
global const v_ref = sqrt(p_ref / rho_ref)
global const E_form_O2 = 0.0
global const E_form_O = 0.5 * E_diss
global const L_ref = 1.0
global const t_ref = L_ref / v_ref
println("tref = $(t_ref)")

global const tmp_e_v_arr = generate_e_vibr_arr_harmonic_K(Θvibr, E_diss)

global const e_int_mol = T -> e_rot_cont(mass_mol, T) + e_vibr_from_array(mass_mol, T, tmp_e_v_arr)
global const c_int_mol = T -> c_rot_cont(mass_mol, T) + c_vibr_from_array(mass_mol, T, tmp_e_v_arr)

global const e_int_atom = T -> 0.0 * T + E_form_O * k_B / mass_atom
global const c_int_atom = T -> 0.0 * T


equations = CompressibleEulerEquationsMs1T2D([mass_mol, mass_atom],
                                             [e_int_mol, e_int_atom],
                                             [c_int_mol, c_int_atom],
                                             mass_ref=m_ref, T_ref=T_ref, e_ref=e_ref,
                                             T_tol=1e-11, ΔT=0.5, min_T_jump_rel=1e-6)

Nx = 6
Ny = Nx
polydeg = 2

initial_condition = initial_condition_weak_blast_wave
init_cond_center = initial_condition_weak_blast_wave([0.0, 0.0], 0.0, equations)
println(cons2prim_scaled(init_cond_center, equations))
init_cond_outerx = initial_condition_weak_blast_wave([1.0, 0.0], 0.0, equations)
println(cons2prim_scaled(init_cond_outerx, equations))
init_cond_outery = initial_condition_weak_blast_wave([0.0, 1.0], 0.0, equations)
println(cons2prim_scaled(init_cond_outery, equations))

volume_flux = flux_oblapenko
surface_flux = flux_oblapenko

solver = DGSEM(polydeg=polydeg, surface_flux=surface_flux,
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-2.0, -2.0)
coordinates_max = ( 2.0,  2.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=Nx,
                n_cells_max=100_000,
                periodicity=true)


outpref = "output/periodic2Dblast/"
outdir = outpref * string(Nx) * "_" * string(Ny) * "/" * string(polydeg) * "/hdf"
outdirvtk = outpref * string(Nx) * "_" * string(Ny) * "/" * string(polydeg) * "/vtk"
mkpath(outdir)
mkpath(outdirvtk)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_condition_periodic)

save_solution = SaveSolutionCallback(dt=0.05,
                                     #interval=250,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim_scaled,
                                     output_directory=outdir)

analysis_callback = AnalysisCallback(semi, interval=10);
callback = CallbackSet(analysis_callback, save_solution)
# ODE solvers
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan);
sol = solve(ode, SSPRK43(); abstol=1.0e-6, reltol=1.0e-6,
            ode_default_options()..., callback=callback);

trixi2vtk(joinpath(outdir, "solution_*.h5"), output_directory=outdirvtk)