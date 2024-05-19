function binary_regulizer_abs(x::T) where {T<:Real}
    abs(abs(x) - one(T))
end
function binary_regulizer_abs(x::AbstractArray{<:Real})
    mean(binary_regulizer_abs.(x))
end


function get_binary_regulizer(pow::Real=1.0)
    # f(x) = abs(1.0 - x^2)^(1/pow)
    f(x) = (abs(1. - x^2)/(1. + abs(x)))^(pow)
    return x -> (typeof(x) <: Float64 && return f(x)) || (return mean(f.(x)))
end

# function get_binary_regulizer(type::String, pow::Real=1.0)
#     if type == "pow"
#         # f(x) = abs(1.0 - x^2)^(1/pow)
#         f(x) = abs(1. - x^2)^(1/pow)
#         return x -> (typeof(x) <: Float64 && return f(x)) || (return mean(f.(x)))
#     elseif type == "abs"
#         return binary_regulizer_abs
#     elseif type == "hyperbolic"
#         f(x) = abs(1. - x^2)^(1/pow)
#         return ternary_regulizer
#     else
#         error("Unsupported type: $type. Choose either 'abs' or 'pow'.")
#     end
# end