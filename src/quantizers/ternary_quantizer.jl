# function ternary_quantizer(x::T) where {T<:Real}
#     x > 0.3 && return one(T)
#     x < -0.3 && return -one(T)
#     return zero(T)
# end

# function ternary_quantizer(x::AbstractArray{<:Real})
#     return ternary_quantizer.(x)
# end

function ternary_quantizer(x::T, t1::Real=-0.5, t2::Real=0.5) where {T<:Real}
    x > t2 && return one(T)
    x < t1 && return -one(T)
    return zero(T)
end

function ternary_quantizer(x::AbstractArray{<:Real}, t1::Real=-0.5, t2::Real=0.5)
    return ternary_quantizer.(x, t1, t2)
end


function ChainRulesCore.rrule(::typeof(ternary_quantizer), x::AbstractMatrix)
    project_x = ProjectTo(x)
    o = ternary_quantizer.(x)
    function ternary_quantizer_pullback(Δy)
        return NoTangent(), Δy
    end
    o, ternary_quantizer_pullback
end

function get_ternary_quantizer(t1::Real=-0.5, t2::Real=0.5)
    return f(x) = ternary_quantizer(x, t1, t2)
end


# function ChainRulesCore.rrule(::typeof(ternary_quantizer), x)
#     y = ternary_quantizer(x)
#     function ternary_quantizer_pullback(ȳ)
#         # parametry -1 1 ternary quantizeru
#         # clamp(ȳ, -1, 1)
#         f̄ = NoTangent()
#         return (f̄, ȳ)
#         # return (f̄, ȳ, NoTangent, NoTangent)
#     end
#     return y, ternary_quantizer_pullback
# end
