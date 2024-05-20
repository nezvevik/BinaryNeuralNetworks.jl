module BTNNs

using Flux
using Flux: Params, create_bias, BatchNorm
using Flux: ChainRulesCore, NoTangent, Chain, params, relu
using Flux: glorot_uniform
using Flux.Data: DataLoader

using Statistics: mean

using ChainRulesCore


include("quantizers/ternary_quantizer.jl")
include("quantizers/binary_quantizer.jl")

include("feature_quantizer.jl")
include("layers/ste_layer.jl")
include("layers/bt_layer.jl")
include("layers/psa_layer.jl")

include("utils.jl")
include("convertors.jl")
include("regularizers.jl")

end # module BTNNs
