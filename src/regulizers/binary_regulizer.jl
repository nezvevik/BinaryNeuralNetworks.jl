function binary_regulizer(x::T) where {T<:Real}
    abs(abs(x) - one(T))
end

# function binary_regulizer(x::T) where {T<:Real}
#     abs(1. - x^2)
# end

function binary_regulizer(x::AbstractArray{<:Real})
    sum(binary_regulizer.(x))
end