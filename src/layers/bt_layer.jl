mutable struct BTLayer{M<:AbstractMatrix,B,F,C,Q1,Q2,R1,R2,N}
    θ::M
    b::B
    σ::F

    weight_compressor::C

    weight_quantizer::Q1
    output_quantizer::Q2

    weight_regularizer::R1
    output_regularizer::R2

    batchnorm::N
end

Flux.@functor BTLayer
function BTLayer(
    input_size::Int,
    output_size::Int,
    σ::Function=identity,
    
    weight_compressor::Function=tanh,

    weight_quantizer::Function=identity,
    output_quantizer::Function=identity,

    weight_regularizer::Function=get_ternary_regularizer(1.0),
    output_regularizer::Function=get_binary_regularizer(1.0),

    batchnorm::Bool=true,
)
    # use Float32
    θ = randn(output_size, input_size)
    b = create_bias(θ, true, size(θ, 1))


    bn = batchnorm ? BatchNorm(size(θ, 1), identity) : identity
    return BTLayer(
        θ, b, σ,
        weight_compressor, weight_quantizer, output_quantizer,
        weight_regularizer, output_regularizer, bn)
end

function DSTLayer(
    in::Integer, out::Integer, σ::Function=identity,
    weight_compressor::Function=tanh, bq::Function=identity, tq::Function=identity)
    return BTLayer(
        in, out,
        bq ∘ weight_compressor, tq ∘ σ)
end

# function PSTLayer(
#     in::Integer, out::Integer, σ::Function=identity,
#     weight_compressor::Function=tanh, bq::Function=identity, tq::Function=identity)
#     return BTLayer(
#         in, out,
#         bq ∘ weight_compressor, tq ∘ σ)
# end



function get_W(l::BTLayer)
    return l.weight_compressor.(l.θ)
end

function get_ternary_W(l::BTLayer)
    return l.weight_quantizer(l.weight_compressor.(l.θ))
end

function (l::BTLayer)(x::VecOrMat)
    W = get_W(l)
    return l.σ.(l.batchnorm(W * x .+ l.b))
end

function (l::BTLayer)(xr::Tuple)
    x, r = xr
    o = l(x)
    r = r + l.output_regularizer(o)
    return (o, r)
end


function set_output_regularizer(l::BTLayer, output_regularizer::Function)
    l.output_regularizer = output_regularizer
end

function set_output_regularizer(m::Chain, β::Function)
    Chain(map(layer -> set_output_regularizer(layer, β), m.layers))
end

function set_weight_regularizer(l::BTLayer, weight_regularizer::Function)
    l.weight_regularizer = weight_regularizer
end

function set_weight_regularizer(m::Chain, τ::Function)
    Chain(map(layer -> set_weight_regularizer(layer, τ), m.layers))
end

