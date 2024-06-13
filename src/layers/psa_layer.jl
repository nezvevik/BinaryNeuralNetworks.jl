

struct PSALayer{M,B,F,Q1,Q2,N}
    W::M
    b::B
    σ::F

    weight_sampler::Q1
    output_sampler::Q2

    batchnorm::N
end

@Flux.functor PSALayer
function PSALayer(
    input_size::Int,
    output_size::Int,
    σ::Function,
    batchnorm::Bool=true
)

    W = randn(output_size, input_size)
    b = create_bias(W, true, size(W, 1))
    return PSALayer(W, b, σ, psa_ternary, psa_binary, batchnorm ? BatchNorm(size(W, 1), identity) : identity)
end


function (l::PSALayer)(x::VecOrMat)
    θ = l.weight_sampler(l.W)
    o = l.σ.(l.batchnorm(θ * x .+ l.b))
    return l.output_sampler(o)
end



function psa_binary(x::Real)
    return(_psa_binary((tanh(x) + 1) / 2))
end 

function psa_binary(x::Matrix)
    return(_psa_binary((@. tanh(x) + 1) / 2))
end 


function _psa_binary(p::T) where {T<:Real}
    rand() > (1-p) ? one(T) : -one(T)
end

function _psa_binary(p::AbstractMatrix)
    _psa_binary.(p)
end

# function ChainRulesCore.rrule(::typeof(_psa_binary), p::Real)
#     println("p")
#     o = _psa_binary(p)
#     function _psa_binary_pullback(Δy)
#         return NoTangent(), 2 * o * Δy
#     end
#     o, _psa_binary_pullback
# end


function ChainRulesCore.rrule(::typeof(_psa_binary), x::AbstractMatrix)
    project_x = ProjectTo(x)
    o = _psa_binary.(x)
    function _psa_binary_pullback(Δy)
        # vase implementace
        # return NoTangent(), -2 .* o .* Δy
        
        # zkopirovano z STE
        # return NoTangent(), 2 .* Δy
        # return NoTangent(), sign.(Δy)
        return NoTangent(), Δy
    end
    o, _psa_binary_pullback
end


psa_ternary(x::Real) = _psa_ternary(tanh(x))
psa_ternary(x::AbstractMatrix) = _psa_ternary(tanh.(x))

function _psa_ternary(x::Real)
    fx = floor(x)
    δ = fx + (_psa_binary(x - fx) + 1) / 2  
end

 function _psa_ternary(x::AbstractMatrix)
    fx = floor.(x)
    δ = fx + (_psa_binary(x - fx) .+ 1) / 2
 end

 
 function convert2discrete(l::PSALayer)
    # RegularizedLayer(l.ρ.(l.W), l.b, binary_quantizer, identity, l.batchnorm)
    BTLayer(l.weight_sampler(copy(l.W)), l.b, l.output_sampler, identity, identity , identity, l.batchnorm)
end

