mutable struct STLayer{M<:AbstractMatrix,B,F,Q1,Q2,N}
    θ::M
    b::B
    σ::F


    weight_quantizer::Q1
    output_quantizer::Q2

    batchnorm::N
end

Flux.@functor STLayer
function STLayer(
    input_size::Int,
    output_size::Int,
    σ::Function=tanh,
    
    weight_quantizer::Function=ternary_quantizer,
    output_quantizer::Function=binary_quantizer,


    batchnorm::Bool=true,
)
    # use Float32
    θ = randn(output_size, input_size)
    b = create_bias(θ, true, size(θ, 1))


    bn = batchnorm ? BatchNorm(size(θ, 1), identity) : identity
    return STLayer(
        θ, b, σ,
        weight_quantizer, output_quantizer, bn)
end

# function STLayer(
#     in::Integer, out::Integer, σ=tanh
# )
#     return STLayer(
#         in,
#         out,
#         σ,
#         tanh,
#         ternary_quantizer,
#         binary_quantizer,
#         true
#     )
# end


function get_W(l::STLayer)
    return l.weight_quantizer(l.θ)
end


function (l::STLayer)(x::VecOrMat)
    W = get_W(l)
    return l.output_quantizer(l.σ.(l.batchnorm(W * x .+ l.b)))
end