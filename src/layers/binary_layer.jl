struct BinaryLayer{M<:AbstractMatrix,B,F,Q,N}
    W::M
    b::B
    σ::F

    q::Q

    batchnorm::N
end

# TODO
# t1 a t2 pro lepsi optimizaci

@Flux.functor BinaryLayer
function BinaryLayer(
    input_size::Int,
    output_size::Int,
    σ::Function,
    batchnorm::Bool=true
)

    W = randn(output_size, input_size)
    b = create_bias(W, true, size(W, 1))
    return BinaryLayer(W, b, σ, ternary_quantizer, batchnorm ? BatchNorm(size(W, 1), σ) : identity)
end

function (l::BinaryLayer)(x::VecOrMat)
    θ = l.q(l.W)
    l.σ.(l.batchnorm(θ * x .+ l.b))
end


function ChainRulesCore.rrule(::typeof(ternary_quantizer), x)
    y = ternary_quantizer(x)
    function ternary_quantizer_pullback(ȳ)
        # parametry -1 1 ternary quantizeru
        # clamp(ȳ, -1, 1)
        f̄ = NoTangent()
        return (f̄, ȳ)
        # return (f̄, ȳ, NoTangent, NoTangent)
    end
    return y, ternary_quantizer_pullback
end


function convert2normal(l::BinaryLayer)
    bin_W = l.q(l.W)
    return BinaryNeuralNetwork.Layer(bin_W, l.b, l.σ, l.batchnorm)
end