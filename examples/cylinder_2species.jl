using Trixi
include("../src/TrixiChem.jl")
using .TrixiChem
using OrdinaryDiffEq
using Trixi2Vtk

function mapping_full(xi_, eta_, cyl_radius, points_shock)

    shock_pos = [(points_shock[1], 0.0), (points_shock[2], points_shock[2]), (0.0, points_shock[3])]  # 3 points that define shock

    # spline has form R[1] + c * eta_01^2 + d * eta_01^3, derivative w.r.t eta_01 is 0 at eta_01 = 0
    R = [sqrt(shock_pos[i][1]^2 + shock_pos[i][2]^2) for i in 1:3]  # 3 radii
    spline_matrix = [1.0 1.0; 0.25 0.125]  # find cubic spline coefficients
    spline_RHS = [R[3] - R[1], R[2] - R[1]]
    spline_cd = spline_matrix \ spline_RHS

    
    eta_01 = (eta_ + 1) / 2
    R_outer = R[1] + spline_cd[1] * eta_01^2 + spline_cd[2] * eta_01^3
    angle = -π/4 + eta_ * π/4

    xi_01 = 0.5 * (-xi_ + 1.0)

    r = (cyl_radius + xi_01 * (R_outer - cyl_radius))

    return SVector(round(r * sin(angle); digits=8), round(r * cos(angle); digits=8))
end

@inline function source_terms_Arrhenius_diss_only(u, x, t,
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

function mapping_flat(xi_, eta_)
    return SVector(xi_, eta_)
end

function mapping(xi, eta)
    x = xi + 0.1 * sin(pi * xi) * sin(pi * eta)
    y = eta + 0.1 * sin(pi * xi) * sin(pi * eta)
    return pi * SVector(x, y)
end

function init_condition_const(x, t, equations::CompressibleEulerEquationsMs1T2D)
    return SVector(0.0, 0.0, 5.0, 0.5, 0.5)
end


@inline function boundary_condition_outflow(u_inner, normal_direction, x, t,
                                                  surface_flux_function, equations::CompressibleEulerEquationsMs1T2D)
    # Calculate the boundary flux entirely from the internal solution state
    # println("outflow, ", orientation, ", ", normal_direction, ", ", x)
    # println(normal_direction, ", ", x)
    flux = Trixi.flux(u_inner, normal_direction, equations)

    return flux
end


@inline function initial_condition_supersonic_flow(x, t, equations::CompressibleEulerEquationsMs1T2D)
    # set the freestream flow parameters
    # rho_freestream = gamma
    # v1 = 10.5
    # v2 = 0.0
    # p_freestream = 1.0
    # prim = SVector(rho_freestream, v1, v2, p_freestream)
    prim = SVector(v1_freestream / v_ref, v2_freestream / v_ref, T_freestream / T_ref, rho_mol_freestream / rho_ref, rho_at_freestream)
    return prim2cons(prim, equations)
end


@inline function boundary_condition_supersonic_inflow(u_inner, normal_direction::AbstractVector, x, t,
    surface_flux_function, equations::CompressibleEulerEquationsMs1T2D)
    u_boundary = initial_condition_supersonic_flow(x, t, equations)

    flux = Trixi.flux(u_boundary, normal_direction, equations)

    return flux
end


# only for P4estMesh{2}
@inline function Trixi.get_boundary_outer_state(u_inner, t,
                                                boundary_condition::typeof(boundary_condition_supersonic_inflow),
                                                normal_direction::AbstractVector,
                                                equations::CompressibleEulerEquationsMs1T2D, dg, cache,
                                                indices...)
    x = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg, indices...)

    return initial_condition_supersonic_flow(x, t, equations)
end


# only for P4estMesh{2}
@inline function Trixi.get_boundary_outer_state(u_inner, t,
                                                boundary_condition::typeof(boundary_condition_outflow),
                                                normal_direction::AbstractVector,
                                                equations::CompressibleEulerEquationsMs1T2D, dg, cache,
                                                indices...)
    return u_inner
end

# only for P4estMesh{2}
@inline function Trixi.get_boundary_outer_state(u_inner, t,
                                                boundary_condition::typeof(TrixiChem.boundary_condition_slip_wall),
                                                normal_direction::AbstractVector,
                                                equations::CompressibleEulerEquationsMs1T2D,
                                                dg, cache, indices...)
    factor = (normal_direction[1] * u_inner[1] + normal_direction[2] * u_inner[2])
    u_normal = (factor / sum(normal_direction .^ 2)) * normal_direction

    return SVector(u_inner[1] - 2 * u_normal[1],
                   u_inner[2] - 2 * u_normal[2],
                   u_inner[3],
                   u_inner[4:end]...)
end

function cons2prim_scaled(u, equations::CompressibleEulerEquationsMs1T2D)
    rho_v1, rho_v2, rho_e, rho_mol, rho_atom = u

    rho = rho_mol + rho_atom

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho

    T = temperature(u, equations)
    p = pressure(u, equations)

    a = sqrt(TrixiChem.get_gamma(u, rho, T, equations) * p / rho)

    return SVector(v1 * v_ref, v2 * v_ref, p * p_ref, T * T_ref, rho_mol * rho_ref, rho_atom * rho_ref, sqrt(v1^2 + v2^2) / a)
end

Trixi.varnames(::typeof(cons2prim_scaled), ::CompressibleEulerEquationsMs1T2D) = ("v1", "v2", "p", "T", "rho_mol", "rho_atom", "M")
global const k_B = 1.380649e-23
global const Θvibr = 3393.48
global const Θvibr_anh = 17.366  # O2
global const E_diss = 59364.8  # O2
global const mass_mol = 5.3134e-26  # O2
global const mass_atom = mass_mol / 2  # O
global const E_form_O2 = 0.0
global const E_form_O = 0.5 * E_diss

global const m_ref = mass_mol
global const cv_ref = k_B / mass_mol


global const v1_freestream = 4000.0
global const v2_freestream = 0.0
global const p_freestream = 500.0
global const T_freestream = 400.0
global const n_freestream = p_freestream / (k_B * T_freestream)
global const x_mol_freestream = 0.9
global const rho_mol_freestream = x_mol_freestream * n_freestream * mass_mol
global const rho_at_freestream = (1.0 - x_mol_freestream) * n_freestream * mass_atom

global const rho_freestream = rho_mol_freestream + rho_at_freestream

global const chem_mult = 1.0

global const A_m_a = 1.6605e-8 * chem_mult
global const n_m_a = -1.5

global const A_m_m = 3.321e-9 * chem_mult
global const n_m_m = -1.5

global const L_ref = 0.045
global const p_ref = p_freestream
global const T_ref = T_freestream
global const n_ref = p_ref / (T_ref * k_B)
global const rho_ref = n_ref * m_ref
global const v_ref = sqrt(p_ref / rho_ref)
global const e_ref = k_B * T_ref / m_ref
global const t_ref = L_ref / v_ref
global const k_scaler = m_ref * n_ref^2 * t_ref / (rho_ref)

println("t_ref = $t_ref")

# global const tmp_e_v_arr = generate_e_vibr_arr_harmonic_K(Θvibr, E_diss)
global const tmp_e_v_arr = generate_e_vibr_arr_anharmonic_K(Θvibr, Θvibr_anh, E_diss) # generate_e_vibr_arr_harmonic_K(Θvibr, E_diss)

global const e_int_mol = T -> e_rot_cont(mass_mol, T) + e_vibr_from_array(mass_mol, T, tmp_e_v_arr)
global const c_int_mol = T -> c_rot_cont(mass_mol, T) + c_vibr_from_array(mass_mol, T, tmp_e_v_arr)

global const e_int_atom = T -> 0.0 * T + E_form_O * k_B / mass_atom
global const c_int_atom = T -> 0.0 * T

equations = CompressibleEulerEquationsMs1T2D([mass_mol, mass_atom],
                                             [e_int_mol, e_int_atom],
                                             [c_int_mol, c_int_atom],
                                             mass_ref=m_ref, T_ref=T_ref, e_ref=e_ref,
                                             T_tol=1e-11, ΔT=0.5, min_T_jump_rel=1e-6)
println(equations)
#
#
#
#                            |
#                            y_pos
#                            |
#                            |
#                            . 
#                          .
#                        .  <- x_neg
#                      .
#  _______y_neg_______.


const boundary_conditions = Dict(:x_neg => boundary_condition_supersonic_inflow,
                                 :y_neg => TrixiChem.boundary_condition_slip_wall,
                                 :y_pos => boundary_condition_outflow,
                                 :x_pos => TrixiChem.boundary_condition_slip_wall)


const Trixi.mapping_full = mapping_full
const polydeg = parse(Int32, ARGS[1])
const cfl = parse(Float64, ARGS[2])

mymapping = (xi, eta) -> Trixi.mapping_full(xi, eta, 1.0, [1.32, 1.05, 2.25])

const surface_flux = FluxLaxFriedrichs(max_abs_speed_naive_new) 
const volume_flux  = FluxRotated(flux_oblapenko)

Nx = 30
Ny = Nx

outpref = "output/2species_cylinder/"
outdir = outpref * string(Nx) * "_" * string(Ny) * "/" * string(polydeg) * "/hdf"
outdirvtk = outpref * string(Nx) * "_" * string(Ny) * "/" * string(polydeg) * "/vtk"
outdirrest = outpref * string(Nx) * "_" * string(Ny) * "/" * string(polydeg) * "/restart"
mkpath(outdir)
mkpath(outdirvtk)
mkpath(outdirrest)


const basis = LobattoLegendreBasis(polydeg)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure)

const local_twosided_variables_cons = ["rho1", "rho2"]
const local_onesided_variables_nonlinear = [(TrixiChem.energy_internal_without_rho, min)]

limiter_idp = SubcellLimiterIDP(equations, basis;
                                local_twosided_variables_cons=local_twosided_variables_cons,
                                local_onesided_variables_nonlinear=local_onesided_variables_nonlinear)
const volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                      volume_flux_dg=volume_flux,
                                                      volume_flux_fv=surface_flux)

trees_per_dimension = (Nx, Ny)
mesh = P4estMesh(trees_per_dimension,
                 polydeg = polydeg, initial_refinement_level = 0,
                 mapping = mymapping,
                 periodicity = (false, false))

solver = DGSEM(polydeg=polydeg, surface_flux=surface_flux,
               volume_integral=volume_integral)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_supersonic_flow, solver,
                                    boundary_conditions=boundary_conditions, source_terms=source_terms_Arrhenius_diss_only)


tspan = (0.0, 2.5)
ode = semidiscretize(semi, tspan)

analysis_callback = AnalysisCallback(semi, interval=100)
alive_callback = AliveCallback(analysis_interval=100)


save_solution = SaveSolutionCallback(dt=0.1,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim_scaled,
                                     output_directory=outdir)

save_restart = SaveRestartCallback(interval=10000, output_directory=outdirrest)


amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max=0.5,
                                          alpha_min=0.001,
                                          alpha_smooth=true,
                                          variable=density_pressure)


amr_controller = ControllerThreeLevel(semi, amr_indicator;
                                      base_level=0,
                                      med_level=1, med_threshold=0.175,
                                      max_level=4, max_threshold=0.35)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=5000,
                           adapt_initial_condition=false,
                           adapt_initial_condition_only_refine=false)

stepsize_callback = StepsizeCallback(cfl = cfl)
callbacks = CallbackSet(analysis_callback, alive_callback, save_solution, save_restart, amr_callback, stepsize_callback)

stage_callbacks = (SubcellLimiterIDPCorrection(),)

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  callback = callbacks);

trixi2vtk(joinpath(outdir, "solution_*.h5"), output_directory=outdirvtk)
