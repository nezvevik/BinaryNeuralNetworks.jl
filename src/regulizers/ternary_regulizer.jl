# weight regulizer
function ternary_regulizer(x::T) where {T<:Real}
    (one(T) - x^2) * x^2
end

# function ternary_regulizer(x::T) where {T<:Real}
#     abs(1-abs(1-2*abs(x)))
# end

function ternary_regulizer(x::AbstractArray{<:Real})
    sum(ternary_regulizer.(x))
end
