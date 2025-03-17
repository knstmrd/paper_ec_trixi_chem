using MuladdMacro
using Trixi
using StaticArrays
using LinearAlgebra

@muladd begin
    struct CompressibleEulerEquationsMs1T2D{NVARS, NCOMP} <:
           Trixi.AbstractCompressibleEulerMulticomponentEquations{2, NVARS, NCOMP} #2 dimensions, NVARS variables, NCOMP components/species
           # NVARS = NCOMP + 3 (NCOMP eqns for density + 2 for velocity and 1 for energy)

           
        mass::SVector{NCOMP, Float64} # molecular mass of each component
        T_ref::Float64
        T_min::Float64
        T_max::Float64
        ΔT::Float64
        inv_ΔT::Float64
        min_T_jump::Float64
        T_tol::Float64

        e_arr::Array{Float64, 2} # Species-specific tabulated energy, N_T_discretization*NCOMP
        c_v_arr::Array{Float64, 2} # Species-specific tabulated specific heat, N_T_discretization*NCOMP
        R_specific::SVector{NCOMP, Float64}  #  k_B / mass [J / kg / K] when not scaled ; component->R_specific

        e_min_arr::SVector{NCOMP, Float64}

        e_ref::Float64 # reference energy
        c_v_ref::Float64 # reference c_v 

        T_arr::Vector{Float64}
        T_arr_inv::Vector{Float64}

        # used to estimate \int c_v(tau) / tau d tau
        int_c_v_over_t_arr::Array{Float64,2} # Species-specific tabulated integral part of entropy, N_T_discretization*NCOMP
    
        # the e_int_function and c_int_function should compute usual dimensional quantities
        # scaling is governed by T_ref, e_ref, c_ref
        function CompressibleEulerEquationsMs1T2D{NVARS, NCOMP}(mass, e_int_function, c_int_function;
                                                  mass_ref=1.0, T_ref=1.0, T_min=10.0, T_max=3.0e4, ΔT=1.0,
                                                  e_ref=1.0, min_T_jump_rel=0.5, T_tol=1e-9) where {NVARS, NCOMP}
            @assert (length(mass)==length(e_int_function)==length(c_int_function)==NCOMP)         

            n_range = trunc(Int, (T_max - T_min) / ΔT) + 1
            T_range = Vector(LinRange(T_min, T_max, n_range))

            @assert abs((T_range[2] - T_range[1]) - ΔT) < 1e-3

            num_comp = size(mass)[1]
            c_v_ref = k_B / mass_ref

            e_arr = map((m, f)->map(t -> 3.0/2.0 * k_B * t / m .+ f(t), T_range), mass, e_int_function)

            e_arr = transpose(stack(e_arr, dims=1))
            println("Size of energy table: ", size(e_arr))

            e_min_arr = minimum(e_arr, dims=1) ./ e_ref
            
            c_v_arr = transpose(stack(map((m, f)->(map(t-> f(t), T_range) .+ ((3.0 / 2.0) * k_B / m)), mass, c_int_function), dims=1))
            
            int_c_v_over_t_arr = zeros(n_range, num_comp)

            #s_int_from_T = T -> c_v_from_T(T) / T
            # we integrate from T_min to T_max using Simpon's rule
            # int_{T_min}^{T_i} = int_{T_min}^{T_{i-1}} + int_{T_{i-1}}^{T_{i}}
            for i in 2:n_range
                T_a = T_range[i-1]
                T_b = T_range[i]
                c_v_a = c_v_arr[i-1, :]
                c_v_b = c_v_arr[i, :]

                int_c_v_over_t_arr[i, :] .= int_c_v_over_t_arr[i-1, :]
                int_c_v_over_t_arr[i, :] .+= (c_v_a - (c_v_b - c_v_a) * T_a / ΔT) * log(T_b / T_a) + (c_v_b - c_v_a)
            end
            T_min /= T_ref
            T_max /= T_ref
            ΔT /= T_ref

            e_arr ./= e_ref

            c_v_arr ./= c_v_ref
            int_c_v_over_t_arr ./= c_v_ref
            mass ./= mass_ref

            T_range ./= T_ref

            new(mass, T_ref, T_min, T_max, ΔT, 1.0 / ΔT, min_T_jump_rel * ΔT,
               T_tol,
               e_arr, c_v_arr, (k_B ./ mass) ./ c_v_ref, e_min_arr, e_ref, c_v_ref, T_range, 1.0 ./ T_range, int_c_v_over_t_arr)
        end
    end

    function CompressibleEulerEquationsMs1T2D(mass, e_int_function, c_int_function;
        mass_ref=1.0, T_ref=1.0, T_min=10.0, T_max=3.0e4, ΔT=1.0,
        e_ref=1.0, min_T_jump_rel=0.5, T_tol=1e-9)
        NCOMP = length(mass)
        NVARS = NCOMP + 3
        return CompressibleEulerEquationsMs1T2D{NVARS,NCOMP}(mass, e_int_function, c_int_function,
                                                                           mass_ref=mass_ref, T_ref=T_ref, T_min=T_min, T_max=T_max,
                                                                           ΔT=ΔT, e_ref=e_ref, min_T_jump_rel=min_T_jump_rel, T_tol=T_tol)
    end


    @inline function Base.real(::CompressibleEulerEquationsMs1T2D{NVARS, NCOMP}) where {NVARS,
                                                                                            NCOMP}
        Float64
    end

    function Trixi.varnames(::typeof(cons2cons),
                    equations::CompressibleEulerEquationsMs1T2D)
        cons = ("rho_v1", "rho_v2", "rho_e")
        rhos = ntuple(n -> "rho" * string(n), Val(ncomponents(equations)))
        return (cons..., rhos...)
    end

    function varnames(::typeof(cons2prim),
        equations::CompressibleEulerEquationsMs1T2D)
        prim = ("v1", "v2", "T")
        rhos = ntuple(n -> "rho" * string(n), Val(ncomponents(equations)))
        return (prim..., rhos...)
    end

    # Convert conservative variables to primitive
    @inline function Trixi.cons2prim(u, equations::CompressibleEulerEquationsMs1T2D)
        rho_v1, rho_v2, rho_e = u 

        prim_rho = SVector{ncomponents(equations), Float64}(u[i + 3]
                                                                    for i in eachcomponent(equations))

        rho = density(u, equations)
        v1 = rho_v1 / rho
        v2 = rho_v2 / rho

        e_internal = (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2)) / rho

        T = temperature(u, e_internal, equations)
        prim_other = SVector{3, Float64}(v1, v2, T)
    
        return vcat(prim_other, prim_rho)
    end

    @inline function Trixi.prim2cons(prim, equations::CompressibleEulerEquationsMs1T2D) 
        (v1, v2, T, rhos...) = prim
        rho=sum(rhos)
        rho_v1 = rho * v1
        rho_v2 = rho * v2
        # rho_e = rho * energy_from_rho_vec(rhos, rho, T, equations) + 0.5 * (rho_v1 * v1 + rho_v2 * v2)
        rho_e = rho * energy_from_rho_vec(rhos, rho, T, equations) + 0.5 * (rho_v1 * v1 + rho_v2 * v2)
        return SVector(rho_v1, rho_v2, rho_e, rhos...)
    end

    @inline function density(u, equations::CompressibleEulerEquationsMs1T2D)
        rho = zero(u[1])
    
        for i in eachcomponent(equations)
            rho += u[i + 3]
        end
    
        return rho
    end

    @inline function density1(u, equations::CompressibleEulerEquationsMs1T2D)
        return u[4]
    end

    @inline function density2(u, equations::CompressibleEulerEquationsMs1T2D)
        return u[5]
    end

    @inline function densities(u, v, equations::CompressibleEulerEquationsMs1T2D)
        return SVector{ncomponents(equations), real(equations)}(u[i + 3] * v
                                                                for i in eachcomponent(equations))
    end

    @inline function density_and_number_density(u, equations::CompressibleEulerEquationsMs1T2D)
        rho = zero(u[1])
        nrho = zero(u[1])
    
        for i in eachcomponent(equations)
            rho += u[i + 3]
            nrho += u[i + 3] / equations.mass[i]
        end
    
        return rho, nrho
    end

    @inline function number_density(u, equations::CompressibleEulerEquationsMs1T2D)
        # rho = zero(u[1])
        nrho = zero(u[1])
    
        for i in eachcomponent(equations)
            #rho += u[i + 3]
            nrho += u[i + 3] / equations.mass[i]
        end
    
        return nrho #rho, nrho
    end

    @inline function pressure(T, u, equations::CompressibleEulerEquationsMs1T2D)
        nrho = number_density(u, equations)
        
        p = T * nrho
        return p
    end

    # State validation for Newton-bisection method of subcell IDP limiting
    @inline function Base.isvalid(u, equations::CompressibleEulerEquationsMs1T2D)
        if u[4] <= 0 || u[5] <= 0
            return false
        end

        eint = energy_internal_without_rho(u, equations)

        if limit_T_low(u, density(u, equations), eint, equations)
            return false
        end
        return true
    end

    @inline function Trixi.pressure(u, equations::CompressibleEulerEquationsMs1T2D)
        rho_v1, rho_v2, rho_e, _ = u
        rho, nrho = density_and_number_density(u, equations)
        e_internal = (rho_e - 0.5 * (rho_v1^2 + rho_v2^2) / rho) / rho

        p = temperature(u, e_internal, equations) * nrho
        return p
    end
    
    @inline function Trixi.density_pressure(u, equations::CompressibleEulerEquationsMs1T2D)
        rho_v1, rho_v2, rho_e, _ = u
        rho, nrho = density_and_number_density(u, equations)
        e_internal = (rho_e - 0.5 * (rho_v1^2 + rho_v2^2) / rho) / rho

        p = temperature(u, e_internal, equations) * nrho
        return rho * p
    end
    
    # internal energy of component i
    @inline function energy_component(i, T, equations::CompressibleEulerEquationsMs1T2D)
        fracpos = (T - equations.T_min) * equations.inv_ΔT #  / equations.ΔT
        
        index_lower = floor(Int, fracpos)
        fracpos -= index_lower
        index_lower += 1
    
        return equations.e_arr[index_lower, i] * (1.0 - fracpos) + fracpos * equations.e_arr[index_lower + 1, i]
    end

    @inline function energy(u::SVector, rho, T, equations::CompressibleEulerEquationsMs1T2D)
        result = 0.0

        for i in eachcomponent(equations)
            result += energy_component(i, T, equations) * u[i + 3] / rho
        end

        return result
    end

    @inline function energy_from_rho_vec(rho_vec::SVector, rho, T, equations::CompressibleEulerEquationsMs1T2D)
        result = 0.0

        for i in eachcomponent(equations)
            result += energy_component(i, T, equations) * rho_vec[i] / rho
        end

        return result
    end

    @inline function energy(u::SVector, T, equations::CompressibleEulerEquationsMs1T2D)
        return energy(u, density(u, equations), T, equations)
    end
    

    @inline function c_v(i, index_lower, fracpos, equations::CompressibleEulerEquationsMs1T2D)
        return equations.c_v_arr[index_lower, i] * (1.0 - fracpos) + fracpos * equations.c_v_arr[index_lower + 1, i]
    end

    # c_v of component i
    @inline function c_v(i, T, equations::CompressibleEulerEquationsMs1T2D)
        fracpos = (T - equations.T_min) * equations.inv_ΔT
        index_lower = floor(Int, fracpos)
        fracpos -= index_lower
        index_lower += 1
    
        return c_v(i, index_lower, fracpos, equations)
    end

    @inline function c_v(u::SVector, rho, T, equations::CompressibleEulerEquationsMs1T2D)
        result = 0.0

        for i in eachcomponent(equations)
            result += c_v(i, T, equations) * u[i + 3] / rho
        end

        return result
    end

    @inline function c_v(u::SVector, T, equations::CompressibleEulerEquationsMs1T2D)
        return c_v(u, density(u, equations), T, equations)
    end

    @inline function limit_T_low(u, rho, e, equations::CompressibleEulerEquationsMs1T2D)
        e_min = 0.0
        for i in eachcomponent(equations)
            e_min += equations.e_min_arr[i] * u[i + 3] / rho
        end
        if e <= e_min
            return true
        else
            return false
        end
    end

    @inline function temperature(u, rho, T0, e, equations::CompressibleEulerEquationsMs1T2D)
        T = T0

        if (T < equations.T_min)
            return 1.0001 * equations.T_min
            # T = 1.0001 * equations.T_min
        elseif (T > equations.T_max)
            return 0.9999 * equations.T_max
        end

        if limit_T_low(u, rho, e, equations)
            return 1.0001 * equations.T_min
        end
        
        fx = energy(u, rho, T, equations) - e
        
        mintol = equations.T_tol * e + equations.T_tol
        
        while abs(fx) > mintol # && iter < 1000  # don't need the second if
            T -= fx / c_v(u, rho, T, equations)   # Iteration
            fx = energy(u, rho, T, equations) - e           # Precompute f(x)
            # iter += 1
        end
        return T
    end

    # probably will not use these, but for completeness' sake
    # compute T(e)
    # E(molecules) = 7/2 T (in scaled form), so 2/7 E as initial guess is not too bad?
    # here e is [J/kg]
    @inline function temperature(u, e, equations::CompressibleEulerEquationsMs1T2D)
        # T0 = 0.28*e
        return temperature(u, density(u, equations), 0.28*e, e, equations)
    end

    @inline function temperature(u, equations::CompressibleEulerEquationsMs1T2D)
        rho = density(u, equations)
        eint = energy_internal(u, equations) / rho

        return temperature(u, rho, 0.28*eint, eint, equations)
    end

     # Calculate total energy for a conservative state `cons`
    @inline energy_total(cons, ::CompressibleEulerEquationsMs1T2D) = cons[3]  #so rhis is just rho_e?    
    
    # Calculate kinetic energy for a conservative state `cons`
    @inline function energy_kinetic(u, rho, equations::CompressibleEulerEquationsMs1T2D)
        rho_v1, rho_v2, _ = u
        #rho = density(u, equations)
        return (rho_v1^2 + rho_v2^2) / (2 * rho)
    end

    @inline function energy_kinetic(u, equations::CompressibleEulerEquationsMs1T2D)
        #  rho_v1, rho_v2, rho_e = u
        rho = density(u, equations)
        return energy_kinetic(u,rho,equations)#(rho_v1^2 + rho_v2^2) / (2 * rho)
    end
     
     # Calculate internal energy for a conservative state `cons`
     @inline function energy_internal(cons, rho, equations::CompressibleEulerEquationsMs1T2D)
        # this returns rho e_internal [J/m^3]
        return energy_total(cons, equations) - energy_kinetic(cons,rho, equations)
    end

    @inline function energy_internal(cons, equations::CompressibleEulerEquationsMs1T2D)
        # this returns rho e_internal [J/m^3]
        return energy_total(cons, equations) - energy_kinetic(cons, equations)
    end
 
    @inline function energy_internal_without_rho(cons, equations::CompressibleEulerEquationsMs1T2D)
        # # this returns e_internal [J/kg]
        rho=density(cons,equations)
        return return energy_internal(cons,rho, equations) / rho
    end

    @inline function entropy_c_v_integral(i::Int64, index_lower, fracpos, T_b, equations::CompressibleEulerEquationsMs1T2D)
        T_a = equations.T_arr[index_lower]
        T_a_inv = equations.T_arr_inv[index_lower]

        c_v_b = c_v(i, index_lower, fracpos, equations)  # value of c_v at T
        c_v_a = equations.c_v_arr[index_lower, i]  # value of c_v at closest_T
        integrate_part = (c_v_a - (c_v_b - c_v_a) * T_a * equations.inv_ΔT) * log(T_b * T_a_inv) + (c_v_b - c_v_a)
    
        return equations.int_c_v_over_t_arr[index_lower, i] + integrate_part
    end
    
    @inline function entropy_c_v_integral(i::Int64, T, equations::CompressibleEulerEquationsMs1T2D)
        fracpos = (T - equations.T_min) * equations.inv_ΔT
        index_lower = floor(Int, fracpos)
        fracpos -= index_lower
        index_lower += 1
        
        return entropy_c_v_integral(i, index_lower, fracpos, T, equations)
    end

    @inline function entropy_c_v_integral(u, T, rho, equations::CompressibleEulerEquationsMs1T2D)
        result = 0.0

        for i in eachcomponent(equations)
            result += entropy_c_v_integral(i, T, equations) * u[i + 3] / rho
        end
        return result
    end

    @inline function entropy_c_v_integral(u, T, equations::CompressibleEulerEquationsMs1T2D)
        rho=density(u, equations)
        return entropy_c_v_integral(u,T,rho, equations)
    end

    @inline function entropy_c_v_integral_without_rho(u, T, equations::CompressibleEulerEquationsMs1T2D)
        result = 0.0

        for i in eachcomponent(equations)
            result += entropy_c_v_integral(i, T, equations) * u[i + 3] 
        end
        return result
    end

    @inline function entropy_thermodynamic(u,rho, equations::CompressibleEulerEquationsMs1T2D)
        
        #e = energy_internal_without_rho(cons, equations)  # get e
        T = temperature(u, equations::CompressibleEulerEquationsMs1T2D)
        s = entropy_c_v_integral(u, T, rho, equations) 
        for i in eachcomponent(equations)
            s -= (u[i + 3] / rho) * log(u[i+3]) / equations.mass[i] 
        end
        return s
    end

    @inline function entropy_thermodynamic(u, equations::CompressibleEulerEquationsMs1T2D)
        
        #e = energy_internal_without_rho(cons, equations)  # get e
        rho=density(u, equations)
        return entropy_thermodynamic(u,rho, equations)
    end

    @inline function entropy_math(u, equations::CompressibleEulerEquationsMs1T2D)
        rho=density(u, equations)
        return -rho * entropy_thermodynamic(u, rho, equations)
    end

    @inline function entropy(u, equations::CompressibleEulerEquationsMs1T2D )
        return entropy_math(u, equations)
    end

    @inline function Trixi.cons2entropy(u, equations::CompressibleEulerEquationsMs1T2D)
        #energy(i, T, equations
        #ncomponents(equations)
        rho_v1, rho_v2, rho_e = u
        #prim_rho = SVector{ncomponents(equations), real(equations)}(u[i + 3]
        #                                                            for i in eachcomponent(equations))
        #rho, nrho = density(u, equations)
        rho = density(u, equations)
        v1 = rho_v1 / rho
        v2 = rho_v2 / rho
        T = temperature(u, equations)
        entr_other = SVector{3, real(equations)}(v1/T, v2/T, -1/T)
        minus_v_half_by_T = -(v1^2 + v2^2)/2/T
        entr_rho = SVector{ncomponents(equations), real(equations)}(-entropy_c_v_integral(i, T,equations) + log(abs(u[i + 3]))/equations.mass[i]
                                                                    + energy_component(i, T, equations)/T + minus_v_half_by_T
                                                                    for i in eachcomponent(equations))
        return vcat(entr_other, entr_rho)
    end

    @inline function Trixi.flux(u, orientation::Integer,
                      equations::CompressibleEulerEquationsMs1T2D)
        rho_v1, rho_v2, rho_e = u

        rho = density(u, equations)

        v1 = rho_v1 / rho
        v2 = rho_v2 / rho

        T = temperature(u, equations)
        p = pressure(T, u, equations)

        if orientation == 1
            f_rho = densities(u, v1, equations)
            f1 = rho_v1 * v1 + p
            f2 = rho_v2 * v1
            f3 = (rho_e + p) * v1
        else
            f_rho = densities(u, v2, equations)
            f1 = rho_v1 * v2
            f2 = rho_v2 * v2 + p
            f3 = (rho_e + p) * v2
        end

        f_other = SVector(f1, f2, f3)

        return vcat(f_other, f_rho)
    end

    # Calculate 1D flux for a single point
    @inline function Trixi.flux(u, normal_direction::AbstractVector,
                        equations::CompressibleEulerEquationsMs1T2D)
        rho_v1, rho_v2, rho_e = u

        rho = density(u, equations)

        v1 = rho_v1 / rho
        v2 = rho_v2 / rho
        v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]

        T = temperature(u, equations)
        p = pressure(T, u, equations)

        f_rho = densities(u, v_normal, equations)
        f1 = rho_v1 * v_normal + p * normal_direction[1]
        f2 = rho_v2 * v_normal + p * normal_direction[2]
        f3 = (rho_e + p) * v_normal

        f_other = SVector(f1, f2, f3)

        return vcat(f_other, f_rho)
    end
    
    @inline function flux_oblapenko(u_ll, u_rr, orientation::Integer,
        equations::CompressibleEulerEquationsMs1T2D)

        (v1_ll, v2_ll, T_ll, rhos_ll...) = cons2prim(u_ll, equations)
        (v1_rr, v2_rr, T_rr, rhos_rr...) = cons2prim(u_rr, equations)

        # this is done just in case at some point some rho < 0.0 and then some logarithm stuff produces a NaN
        # but if we take (abs(rho)) then we might avoid a full breakdown of the simulation and get to a physically meaningful result
        rhos_ll = abs.(rhos_ll)
        rhos_rr = abs.(rhos_rr)

        v1_avg = 0.5*(v1_ll + v1_rr)
        v2_avg = 0.5*(v2_ll + v2_rr)
        inv_T_avg = 0.5 * (1.0 / T_ll + 1.0 / T_rr)
        T_geo_sqr = T_ll * T_rr

        velocity_square_avg = 0.5 * (v1_ll^2 + v2_ll^2 + v1_rr^2 + v2_rr^2)
        T_jump = T_rr - T_ll

        tmp_sum = 0.0
        for i in eachcomponent(equations)
            tmp_sum+=0.5*((rhos_ll[i]+rhos_rr[i])/equations.mass[i])
        end

        if(orientation == 1)
            fx_rhos = SVector{ncomponents(equations), Float64}(Trixi.ln_mean(rhos_ll[i], rhos_rr[i]) * v1_avg
                                                               for i in eachcomponent(equations))  #use ln_mean function in math.jl
            fx_rhos_sum = sum(fx_rhos)                                      
            fx_rho_v1 = v1_avg * fx_rhos_sum  + tmp_sum / inv_T_avg
            fx_rho_v2 = v2_avg * fx_rhos_sum
            fx_e = v1_avg * fx_rho_v1 + v2_avg * fx_rho_v2 - 0.5 * fx_rhos_sum * velocity_square_avg

            if (abs(T_jump) < equations.min_T_jump)
                T_mid = 0.5 * (T_ll + T_rr)
                inv_T_mid = 1.0 / T_mid
                for i in eachcomponent(equations)
                    cvmid = c_v(i, T_mid, equations)        
                    fx_e += fx_rhos[i] * (0.5*(energy_component(i, T_ll, equations) + energy_component(i, T_rr, equations)) + T_geo_sqr * (cvmid  * inv_T_mid - inv_T_avg * cvmid))
                end
            else
                inv_T_jump = 1.0 / T_jump
                for i in eachcomponent(equations)
                    cv_Tast_over_Tast = (entropy_c_v_integral(i, T_rr, equations) - entropy_c_v_integral(i, T_ll, equations)) * inv_T_jump

                    e_int_ll = energy_component(i, T_ll, equations) 
                    e_int_rr = energy_component(i, T_rr, equations) 
                    cv_T_astast = (e_int_rr - e_int_ll) * inv_T_jump
                    
                    fx_e += fx_rhos[i] * (0.5*(e_int_ll+e_int_rr) + T_geo_sqr * (cv_Tast_over_Tast - inv_T_avg* cv_T_astast))
                end
            end
        else
            fx_rhos = SVector{ncomponents(equations), Float64}(Trixi.ln_mean(rhos_ll[i], rhos_rr[i]) * v2_avg
                                                                for i in eachcomponent(equations))
            fx_rhos_sum = sum(fx_rhos)                                      
            fx_rho_v2 = v2_avg * fx_rhos_sum  + tmp_sum / inv_T_avg
            fx_rho_v1 = v1_avg * fx_rhos_sum
            fx_e = v1_avg * fx_rho_v1 + v2_avg * fx_rho_v2 - 0.5 * fx_rhos_sum * velocity_square_avg
            if (abs(T_jump) < equations.min_T_jump)
                T_mid = 0.5 * (T_ll + T_rr)
                inv_T_mid = 1.0 / T_mid
                for i in eachcomponent(equations)
                    cvmid = c_v(i, T_mid, equations)
                    fx_e += fx_rhos[i] * (0.5*(energy_component(i, T_ll, equations) + energy_component(i, T_rr, equations)) + T_geo_sqr * (cvmid  * inv_T_mid - inv_T_avg * cvmid))
                end
            else
                inv_T_jump = 1.0  / T_jump
                for i in eachcomponent(equations)
                    cv_Tast_over_Tast = (entropy_c_v_integral(i, T_rr, equations) - entropy_c_v_integral(i, T_ll, equations)) * inv_T_jump

                    e_int_ll = energy_component(i, T_ll, equations) 
                    e_int_rr = energy_component(i, T_rr, equations) 
                    cv_T_astast = (e_int_rr - e_int_ll) * inv_T_jump

                    fx_e += fx_rhos[i] * (0.5*(e_int_ll+e_int_rr) + T_geo_sqr * (cv_Tast_over_Tast - inv_T_avg* cv_T_astast))
                end
            end
        end
        return SVector(fx_rho_v1, fx_rho_v2, fx_e, fx_rhos...)
    end


    # Called inside `FluxRotated` in `numerical_fluxes.jl` so the direction
    # has been normalized prior to this rotation of the state vector
    @inline function Trixi.rotate_to_x(u, normal_vector, equations::CompressibleEulerEquationsMs1T2D)
        # cos and sin of the angle between the x-axis and the normalized normal_vector are
        # the normalized vector's x and y coordinates respectively (see unit circle).
        c = normal_vector[1]
        s = normal_vector[2]
    
        # Apply the 2D rotation matrix with normal and tangent directions of the form
        # [ 1    0    0   0;
        #   0   n_1  n_2  0;
        #   0   t_1  t_2  0;
        #   0    0    0   1 ]
        # where t_1 = -n_2 and t_2 = n_1
        densities = @view u[4:end]

        return SVector(c * u[1] + s * u[2],
                       -s * u[1] + c * u[2],
                       u[3],
                       densities...)
    end
    
    # Called inside `FluxRotated` in `numerical_fluxes.jl` so the direction
    # has been normalized prior to this back-rotation of the state vector
    @inline function Trixi.rotate_from_x(u, normal_vector,
                                   equations::CompressibleEulerEquationsMs1T2D)
        c = normal_vector[1]
        s = normal_vector[2]
    
        densities = @view u[4:end]
        return SVector(c * u[1] - s * u[2],
                       s * u[1] + c * u[2],
                       u[3],
                       densities...)
    end


    @inline function get_gamma(u, rho, T, equations::CompressibleEulerEquationsMs1T2D)
        c_v_val = c_v(u, rho, T, equations)
        c_p = 0.0

        # c_p = c_v + \sum_i rho_i k/m_i/rho =(scaling)= \sum_i rho_i' 1.0/m_i'/rho' (' denotes scaled variables)
        for i in eachcomponent(equations)
            c_p += u[i + 3] / equations.mass[i] / rho
        end

        return (c_v_val + c_p) / c_v_val
    end

    @inline function Trixi.max_abs_speeds(u, equations::CompressibleEulerEquationsMs1T2D)
        (v1, v2, T, rhos...) = cons2prim(u, equations)
        rho = sum(rhos)
        # c = sqrt(gamma * p / rho)
        gamma = get_gamma(u, rho, T, equations)
        p = pressure(T, u, equations)
        
        c = sqrt(gamma * p / rho)
        return abs(v1) + c, abs(v2) + c
    end
    
    @inline function max_abs_speed_naive_new(u_ll, u_rr, normal_direction::AbstractVector,
                                         equations::CompressibleEulerEquationsMs1T2D)


        (v1_ll, v2_ll, T_ll, rhos_ll...) = cons2prim(u_ll, equations)
        (v1_rr, v2_rr, T_rr, rhos_rr...) = cons2prim(u_rr, equations)

        rho_ll = sum(rhos_ll)
        rho_rr = sum(rhos_rr)

        gamma_ll = get_gamma(u_ll, rho_ll, T_ll, equations)
        gamma_rr = get_gamma(u_rr, rho_rr, T_rr, equations)
    
        p_ll = pressure(T_ll, u_ll, equations)
        p_rr = pressure(T_rr, u_rr, equations)
        # Calculate normal velocities and sound speed
        # left
        v_ll = (v1_ll * normal_direction[1]
                +
                v2_ll * normal_direction[2])
        # c_ll = sqrt(gamma_ll * p_ll / rho_ll)
        c_ll = sqrt(gamma_ll * p_ll / rho_ll)
        # right
        v_rr = (v1_rr * normal_direction[1]
                +
                v2_rr * normal_direction[2])
        # c_rr = sqrt(gamma_rr * p_rr / rho_rr)
        c_rr = sqrt(gamma_rr * p_rr / rho_rr)
    
        norm_norm = norm(normal_direction)
        return max(abs(v_ll) + c_ll * norm_norm, abs(v_rr) + c_rr * norm_norm)
    end

    """
        boundary_condition_slip_wall(u_inner, normal_direction, x, t, surface_flux_function,
                                     equations::CompressibleEulerEquationsMs1T2D)
    
    Determine the boundary numerical surface flux for a slip wall condition.
    Imposes a zero normal velocity at the wall.
    Density is taken from the internal solution state and pressure is computed as an
    exact solution of a 1D Riemann problem. Further details about this boundary state
    are available in the paper:
    - J. J. W. van der Vegt and H. van der Ven (2002)
      Slip flow boundary conditions in discontinuous Galerkin discretizations of
      the Euler equations of gas dynamics
      [PDF](https://reports.nlr.nl/bitstream/handle/10921/692/TP-2002-300.pdf?sequence=1)
    
    Details about the 1D pressure Riemann solution can be found in Section 6.3.3 of the book
    - Eleuterio F. Toro (2009)
      Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction
      3rd edition
      [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)
    
    Should be used together with [`UnstructuredMesh2D`](@ref).
    """
    @inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector,
                                                  x, t,
                                                  surface_flux_function,
                                                  equations::CompressibleEulerEquationsMs1T2D)

        norm_ = norm(normal_direction)
        # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
        normal = normal_direction / norm_
    
        # rotate the internal solution state
        u_local = Trixi.rotate_to_x(u_inner, normal, equations)
    
        # compute the primitive variables
        # rho_local, v_normal, v_tangent, p_local, T_local = cons2prim(u_local, equations)
        (v_normal, v_tangent, T_local, rhos_local...) = cons2prim(u_local, equations)
        rho_local = sum(rhos_local)
        gamma_local = get_gamma(u_local, rho_local, T_local, equations)
    
        p_local = pressure(T_local, u_local, equations)
        # Get the solution of the pressure Riemann problem
        # See Section 6.3.3 of
        # Eleuterio F. Toro (2009)
        # Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction
        # [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)
        if v_normal <= 0.0
            sound_speed = sqrt(gamma_local * p_local / rho_local) # local sound speed
            # try
            p_star = p_local *
                     (1 + 0.5 * (gamma_local - 1) * v_normal / sound_speed)^(2 *
                                                                            gamma_local / (gamma_local - 1))
        else # v_normal > 0.0
            A = 2 / ((gamma_local + 1) * rho_local)
            B = p_local * (gamma_local - 1) / (gamma_local + 1)
            p_star = p_local +
                     0.5 * v_normal / A *
                     (v_normal + sqrt(v_normal^2 + 4 * A * (p_local + B)))
        end
    
        # For the slip wall we directly set the flux as the normal velocity is zero
        return SVector(p_star * normal[1],
                       p_star * normal[2],
                       0.0,
                       zeros(ncomponents(equations))...) * norm_
    end
    
    """
        boundary_condition_slip_wall(u_inner, orientation, direction, x, t,
                                     surface_flux_function, equations::CompressibleEulerEquations2D)
    
    Should be used together with [`TreeMesh`](@ref).
    """
    @inline function boundary_condition_slip_wall(u_inner, orientation,
                                                  direction, x, t,
                                                  surface_flux_function,
                                                  equations::CompressibleEulerEquationsMs1T2D)

        # get the appropriate normal vector from the orientation
        if orientation == 1
            normal_direction = SVector(1, 0)
        else # orientation == 2
            normal_direction = SVector(0, 1)
        end
    
        # compute and return the flux using `boundary_condition_slip_wall` routine above
        return boundary_condition_slip_wall(u_inner, normal_direction, direction,
                                            x, t, surface_flux_function, equations)
    end
    
    """
        boundary_condition_slip_wall(u_inner, normal_direction, direction, x, t,
                                     surface_flux_function, equations::CompressibleEulerEquations2D)
    
    Should be used together with [`StructuredMesh`](@ref).
    """
    @inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector,
                                                  direction, x, t,
                                                  surface_flux_function,
                                                  equations::CompressibleEulerEquationsMs1T2D)
        # flip sign of normal to make it outward pointing, then flip the sign of the normal flux back
        # to be inward pointing on the -x and -y sides due to the orientation convention used by StructuredMesh
        if isodd(direction)
            boundary_flux = -boundary_condition_slip_wall(u_inner, -normal_direction,
                                                          x, t, surface_flux_function,
                                                          equations)
        else
            boundary_flux = boundary_condition_slip_wall(u_inner, normal_direction,
                                                         x, t, surface_flux_function,
                                                         equations)
        end
    
        return boundary_flux
    end
end # @muladd
