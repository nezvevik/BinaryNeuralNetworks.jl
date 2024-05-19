struct BTLayer{M<:AbstractMatrix,B,F,C,Q1,Q2,R1,R2,N}
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
    θ::AbstractArray,
    b::AbstractArray=create_bias(θ, true, size(θ, 1)),
    σ::Function=identity,
    
    weight_compressor::Function=identity,
    weight_quantizer::Function=identity,
    output_quantizer::Function=identity,

    weight_regulizer::Function=ternary_regulizer,
    output_regularizer::Function=binary_regulizer_abs,

    batchnorm::Function=BatchNorm(size(θ, 1), identity),
)

    return BTLayer(
        θ, b, σ,
        weight_compressor, weight_quantizer, output_quantizer,
        weight_regulizer, output_regularizer, batchnorm)
end

Flux.@functor BTLayer
function BTLayer(
    input_size::Int,
    output_size::Int,
    σ::Function=identity,
    
    weight_compressor::Function=identity,
    weight_quantizer::Function=identity,
    output_quantizer::Function=identity,

    weight_regulizer::Function=ternary_regulizer,
    output_regularizer::Function=binary_regulizer_abs,

    batchnorm::Bool=true,
)
    # use Float32
    θ = randn(output_size, input_size)
    b = create_bias(θ, true, size(θ, 1))


    bn = batchnorm ? BatchNorm(size(θ, 1), identity) : identity
    return BTLayer(
        θ, b, σ,
        weight_compressor, weight_quantizer, output_quantizer,
        weight_regulizer, output_regularizer, bn)
end

function get_W(l::BTLayer)
    return l.weight_compressor.(l.θ)
end

function get_ternary_W(l::BTLayer)
    return l.weight_quantizer(l.weight_compressor.(l.θ))
end

function (l::BTLayer)(x::VecOrMat)
    W = get_W(l)
    o = l.σ.(l.batchnorm(W * x .+ l.b))
    l.output_quantizer.(o)
end

function (l::BTLayer)(xr::Tuple)
    x, r = xr
    o = l(x)
    r = r + l.output_regularizer(o)
    return (o, r)
end

# Regularizers
function weight_regulizer(l::BTLayer)
    l.weight_regularizer(get_W(l))
end

# activation regulizer
function activation_regulizer(m::Union{Chain, BTLayer}, x::VecOrMat)
    _, r = m((x, 0.0))
    return r
end

function activation_regulizer(m::Union{Chain, BTLayer}, data::DataLoader)
    r = 0f0
    for (x, _) in data
        r += activation_regulizer(m, x)
    end
    return r / (length(data))
end



function set_output_regularizer(l::BTLayer, β::Function)
    BTLayer(l.W, l.b, l.σ, l.ρ, β, l.τ, l.batchnorm)
end

function set_output_regularizer(m::Chain, β::Function)
    Chain(map(layer -> set_output_regularizer(layer, β), m.layers))
end

function set_weight_regularizer(l::BTLayer, τ::Function)
    BTLayer(l.W, l.b, l.σ, l.ρ, l.β, τ, l.batchnorm)
end

function set_weight_regularizer(m::Chain, τ::Function)
    Chain(map(layer -> set_weight_regularizer(layer, τ), m.layers))
end


# convert network
function convert2discrete(m::Chain)
    Chain(map(convert2discrete, m.layers))
end

function convert2discrete(l::BTLayer)
    BTLayer(get_ternary_W(l), l.b, binary_quantizer, identity, l.β, l.τ, l.batchnorm)
end


function convert2binary_activation(m::Chain)
    Chain(map(convert2binary_activation, m.layers))
end

# ternary weights
function convert2ternary_weights(m::Chain)
    Chain(map(convert2ternary_weights, m.layers))
end

function convert2ternary_weights(l::BTLayer)
    BTLayer(copy(get_ternary_W(l)), l.b, l.σ, identity, l.β, l.τ, l.batchnorm)
end

function convert2binary_activation(l::BTLayer)
    BTLayer(l.ρ.(l.W), l.b, binary_quantizer, identity, l.β, l.τ, l.batchnorm)
end



