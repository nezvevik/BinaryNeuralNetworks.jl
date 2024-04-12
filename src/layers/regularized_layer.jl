struct RegularizedLayer{M<:AbstractMatrix,B,F,R,N}
    W::M
    b::B
    σ::F

    ρ::R

    batchnorm::N
end

Flux.@functor RegularizedLayer

function RegularizedLayer(
    input_size::Int,
    output_size::Int,
    σ::Function,
    ρ::Function=identity,
    batchnorm::Bool=true,
)
    # use Float32
    W = randn(output_size, input_size)
    b = create_bias(W, true, size(W, 1))
    return RegularizedLayer(W, b, σ, ρ, batchnorm ? BatchNorm(size(W, 1), σ) : identity)
end

function (l::RegularizedLayer)(x::VecOrMat)
    θ = l.ρ.(l.W)
    l.σ.(l.batchnorm(θ * x .+ l.b))
end

function (l::RegularizedLayer)(xr::Tuple)
    x, r = xr
    o = l(x)
    r = r + binary_regulizer(o)
    return (o, r)
end


# weight regulizer
function weight_regulizer(l::RegularizedLayer)
    ternary_regulizer(l.ρ.(l.W))
end

# activation regulizer
function activation_regulizer(m::Union{Chain, RegularizedLayer, Layer}, data::DataLoader)
    r = 0f0
    for (x, _) in data
        r += activation_regulizer(m, x)
    end
    return r
end

function activation_regulizer(m::Union{Chain, RegularizedLayer, Layer}, x::VecOrMat)
    _, r = m((x, 0.0))
    return r
end

# convert network
function convert2discrete(m::Chain)
    Chain(map(convert2discrete, m.layers))
end

function convert2binary_activation(m::Chain)
    Chain(map(convert2binary_activation, m.layers))
end

function convert2ternary_weights(m::Chain)
    Chain(map(convert2ternary_weights, m.layers))
end

function convert2binary_activation(l::RegularizedLayer)
    RegularizedLayer(l.ρ.(l.W), l.b, binary_quantizer, identity, l.batchnorm)
end

function convert2ternary_weights(l::RegularizedLayer)
    RegularizedLayer(ternary_quantizer(l.ρ.(l.W)), l.b, l.σ, identity, l.batchnorm)
end

function convert2discrete(l::RegularizedLayer)
    # RegularizedLayer(l.ρ.(l.W), l.b, binary_quantizer, identity, l.batchnorm)
    RegularizedLayer(ternary_quantizer(l.ρ.(l.W)), l.b, binary_quantizer, identity, l.batchnorm)
end