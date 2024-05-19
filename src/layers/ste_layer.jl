struct STELayer{M<:AbstractMatrix,B,F,Q1,Q2,T1,T2,N}
    W::M
    b::B
    σ::F

    weight_quantizer::Q1
    output_quantizer::Q2

    t1::T1
    t2::T2

    batchnorm::N
end

# TODO
# t1 a t2 pro lepsi optimizaci

@Flux.functor STELayer
function STELayer(
    input_size::Int,
    output_size::Int,
    σ::Function,
    batchnorm::Bool=true
)

    W = randn(output_size, input_size)
    b = create_bias(W, true, size(W, 1))

    t1 = -0.5
    t2 = 0.5

    # f = get_tern_quantizer(t1, t2)
    f = ternary_quantizer

    return STELayer(
        W, b, σ, f, binary_quantizer,
        t1, t2, 
        batchnorm ? BatchNorm(size(W, 1), identity) : identity)
end


function (l::STELayer)(x::VecOrMat)
    θ = l.weight_quantizer(tanh.(l.W))
    l.output_quantizer(l.σ.(l.batchnorm(θ * x .+ l.b)))
end

# function (l::STELayer)(x::VecOrMat)
#     θ = l.q(tanh.(l.W))
#     l.σ.(l.batchnorm(θ * x .+ l.b))
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


# function convert2normal(l::STELayer)
#     bin_W = l.q(l.W)
#     return BinaryNeuralNetwork.Layer(bin_W, l.b, l.σ, l.batchnorm)
# end

function ChainRulesCore.rrule(::typeof(binary_quantizer), x::AbstractMatrix)
    project_x = ProjectTo(x)
    o = binary_quantizer.(x)
    function binary_quantizer_pullback(Δy)
        return NoTangent(), Δy
    end
    o, binary_quantizer_pullback
end
