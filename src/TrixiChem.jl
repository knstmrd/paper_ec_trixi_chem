module TrixiChem

const k_B::Float64 = 1.380649e-23  # J / K

include("energy_utils.jl")
include("compressible_euler_2d_ms1T.jl")

export flux_oblapenko
export k_B, e_rot_cont, e_vibr_from_array, e_vibr_iho
export c_rot_cont, c_vibr_from_array, c_vibr_iho
export generate_e_vibr_arr_harmonic_K, generate_e_vibr_arr_anharmonic_K
export max_abs_speed_naive_new, max_abs_speed_naive
export flux_lax_friedrichs_es, FluxLaxFriedrichsEs
export CompressibleEulerEquationsMs1T2D
export prim2cons, cons2prim
export temperature
export boundary_condition_slip_wall

end