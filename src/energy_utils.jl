# compute rotational energy assuming a continuous and fully excited spectrum
function e_rot_cont(m, T)
    return k_B * T / m
end

function c_rot_cont(m, T)
    return k_B / m
end

# compute non-scaled vibrational energy, infinite harmonic oscillator, J / kg
function e_vibr_iho(m, Θ, T)
    return (k_B / m) * Θ ./ ((exp.(Θ./T) .- 1.0))
end

# compute non-scaled vibrational specific heat, infinite harmonic oscillator, J/kg/K
function c_vibr_iho(m, Θ, T)
    return (k_B / m) * (Θ ./ T).^2 .* exp.(Θ ./ T) ./ (exp.(Θ ./ T) .- 1.0).^2
end

function generate_e_vibr_arr_harmonic_K(Θ, E_diss)
    i_max = trunc(Int, (E_diss / Θ - 0.5))  # find maximum level before vibrational energy exceeds dissociation energy, (i+0.5) theta < E_diss_K
    return collect(0:i_max) .* Θ  # units are K
end

function generate_e_vibr_arr_anharmonic_K(Θ, Θ_anh, E_diss)
    v_e_arr = []
    eold = -1.0
    i = 0
    while(true)
        enew = (i + 0.5) * Θ - (i + 0.5)^2 * Θ_anh
        if (enew < E_diss)
            eold = enew
            push!(v_e_arr, enew)
            i += 1
        else
            break
        end
    end
    return v_e_arr .-= v_e_arr[1]
end

function Z_vibr(T, E_vibr_array_K)
    return sum(exp.(-E_vibr_array_K ./ T))
end

function avg_vibr_array(T, E_vibr_array_K, array_to_avg)
    Z_v = Z_vibr(T, E_vibr_array_K)

    return sum(array_to_avg .* exp.(-E_vibr_array_K ./ T)) / Z_v
end

function e_vibr_from_array(m, T, E_vibr_array_K)
    Z_v = Z_vibr(T, E_vibr_array_K)

    return (k_B / m) * avg_vibr_array(T, E_vibr_array_K, E_vibr_array_K)
end

# compute non-scaled vibrational specific heat, cut-off harmonic oscillator, J/kg/K
function c_vibr_from_array(m, T, E_vibr_array_K)
    avg_e_sq = avg_vibr_array(T, E_vibr_array_K, E_vibr_array_K .^ 2)
    avg_e = avg_vibr_array(T, E_vibr_array_K, E_vibr_array_K)
    return (k_B / m) * (avg_e_sq - avg_e^2) / T^2
end