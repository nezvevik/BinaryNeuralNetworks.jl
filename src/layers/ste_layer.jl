struct STELayer{M<:AbstractMatrix,B,F,Q1,Q2,N}
    W::M
    b::B
    σ::F

    weight_quantizer::Q1
    output_quantizer::Q2


    batchnorm::N
end


@Flux.functor STELayer
function STELayer(
    input_size::Int,
    output_size::Int,
    σ::Function,
    batchnorm::Bool=true
)

    W = randn(output_size, input_size)
    b = create_bias(W, true, size(W, 1))


    f = ternary_quantizer

    return STELayer(
        W, b, σ, f, binary_quantizer,
        batchnorm ? BatchNorm(size(W, 1), identity) : identity)
end


function (l::STELayer)(x::VecOrMat)
    θ = l.weight_quantizer(tanh.(l.W))
    l.output_quantizer(l.σ.(l.batchnorm(θ * x .+ l.b)))
end


