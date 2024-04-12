function binary_quantizer(x::T) where T<:Real
    x == zero(T) && return one(T)
    return sign(x)
end

function binary_quantizer(x::AbstractArray{T}) where T
    binary_quantizer.(x)
end