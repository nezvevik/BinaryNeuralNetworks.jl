struct Layer{M<:AbstractMatrix,B,F,N}
    W::M
    b::B
    σ::F

    batchnorm::N
end


function Layer(
    input_size::Int,
    output_size::Int,
    σ::Function,
    batchnorm::Bool=true
)

    W = randn(output_size, input_size)
    b = create_bias(W, true, size(W, 1))
    return Layer(W, b, σ, batchnorm ? BatchNorm(size(W, 1), σ) : identity)
end


function (l::Layer)(x::VecOrMat)
    l.σ.(l.batchnorm(l.W * x .+ l.b))
end

function (l::Layer)(xr::Tuple)
    x, r = xr
    o = l(x)
    r = r + binary_regulizer(o)
    return (o, r)
end

function Flux.params(l::Layer)
    return Params([l.W, l.b])
end

# weight regulizer
function weight_regulizer(m::Chain)
    mapreduce(weight_regulizer, +, m.layers)
end

function weight_regulizer(l::Layer)
    weight_regulizer(l.W)
end

# activation regulizer
# function activation_regulizer(l::Layer)
#     activation_regulizer(l.W)
# end


function Flux.params(c::Flux.Chain)
    θ = []
    for l in c.layers
        push!(θ, Flux.params(l)...)
    end
    return Params(θ)
end

# creates a normal neural network with binary weights
function convert2binary(m::Chain)
    Chain(map(convert2binary, m.layers))
end

function convert2binary_activation(m::Chain)
    Chain(map(convert2binary_activation, m.layers))
end

function convert2binary_weights(m::Chain)
    Chain(map(convert2binary_weights, m.layers))
end
