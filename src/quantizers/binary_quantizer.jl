function binary_quantizer(x::T) where T<:Real
    x < zero(T) && return -one(T)
    return one(T)
end

function binary_quantizer(x::AbstractArray{T}) where T
    binary_quantizer.(x)
end

function ChainRulesCore.rrule(::typeof(binary_quantizer), x)
    println("jsem tady")
    y = binary_quantizer(x)
    function binary_quantizer_pullback(ȳ)
        # parametry -1 1 ternary quantizeru
        # clamp(ȳ, -1, 1)
        f̄ = NoTangent()
        return (f̄, ȳ)
        # return (f̄, ȳ, NoTangent, NoTangent)
    end
    return y, binary_quantizer_pullback
end

function ChainRulesCore.rrule(::typeof(binary_quantizer), x::AbstractMatrix)
    project_x = ProjectTo(x)
    o = binary_quantizer.(x)
    function binary_quantizer_pullback(Δy)
        return NoTangent(), Δy
    end
    o, binary_quantizer_pullback
end
