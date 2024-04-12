function ternary_quantizer(x::T) where {T<:Real}
    x > 0.5 && return one(T)
    x < -0.5 && return -one(T)
    return zero(T)
end

function ternary_quantizer(x::AbstractArray{<:Real})
    return ternary_quantizer.(x)
end