function ternary_regulizer(x::T) where {T<:Real}
    abs(one(T) - abs(one(T) - 2 * abs(x)))
end

function ternary_regulizer(x::AbstractArray{<:Real})
    mean(ternary_regulizer.(x))
end


function get_ternary_regulizer(pow::Real=1.0)
    # f(x) = abs(1.0 - x^2)^(1/pow)
    f(x) = abs(4*(abs(x) - x*x)/(1 + abs(1 - 2*abs(x))))^pow
    return x -> (typeof(x) <: Float64 && return f(x)) || (return mean(f.(x)))
end

