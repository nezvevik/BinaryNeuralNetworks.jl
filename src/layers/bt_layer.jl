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

    weight_quantizer::Function=get_ternary_quantizer(-0.5,0.5),
    output_quantizer::Function=binary_quantizer,

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

function RLayer(
    in::Integer, out::Integer, σ::Function=tanh, weight_compressor::Function=tanh,
    wr::Function=get_ternary_regularizer(1.0), or::Function=get_binary_regularizer(1.0))
    return BTLayer(
        in, out, σ,
        weight_compressor, get_ternary_quantizer(-0.5,0.5), binary_quantizer,
        wr, or)
end

# function DSTLayer(
#     in::Integer, out::Integer, σ::Function=identity,
#     wc::Function=tanh, bq::Function=binary_quantizer, tq::Function=ternary_quantizer)
#     # return BTLayer(
#     #     in, out,
#     #     bq ∘ σ, tq ∘ wc)
#     f1(x) = bq(σ(x))
#     f2(x) = tq(wc(x))
#     return BTLayer(
#         in, out,
#         f1, f2)
# end

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


function set_output_regularizer(l::BTLayer, or::Function)
    l.output_regularizer = or
end

function set_output_regularizer(m::Chain, or::Function)
    Chain(map(layer -> set_output_regularizer(layer, or), m.layers))
end

function set_weight_regularizer(l::BTLayer, wr::Function)
    l.weight_regularizer = wr
end

function set_weight_regularizer(m::Chain, wr::Function)
    Chain(map(layer -> set_weight_regularizer(layer, wr), m.layers))
end

function set_weight_quantizer(l::BTLayer, wq::Function)
    l.weight_quantizer = wq
end

function set_weight_quantizer(m::Chain, wq::Function)
    Chain(map(layer -> set_weight_quantizer(layer, wq), m.layers))
end

