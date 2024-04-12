struct Layer{M<:AbstractMatrix,B,F,N}
    W::M
    b::B
    σ::F

    batchnorm::N
end

Flux.@functor Layer
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
    ternary_regulizer(l.W)
end

# activation regulizer
# function activation_regulizer(l::Layer)
#     activation_regulizer(l.W)
# end


# creates a normal neural network with binary weights

