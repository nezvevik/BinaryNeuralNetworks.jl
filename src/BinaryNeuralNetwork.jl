module BinaryNeuralNetwork

using Flux
using Flux: Params, create_bias, BatchNorm
using Flux: ChainRulesCore, NoTangent, Chain, params, relu
using Flux: glorot_uniform
using Flux.Data: DataLoader

using Statistics: mean

using ChainRulesCore

include("layers/feature_quantizer.jl")

include("regulizers/binary_regulizer.jl")
include("regulizers/ternary_regulizer.jl")

include("quantizers/ternary_quantizer.jl")
include("quantizers/binary_quantizer.jl")

include("layers/binary_layer.jl")
include("layers/regularized_layer.jl")
include("layers/psa_layer.jl")

include("utils.jl")

end # module BinaryNeuralNetwork
