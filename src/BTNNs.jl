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

include("layers/bt_layer.jl")
include("layers/psa_layer.jl")
include("layers/st_layer.jl")


include("convertors.jl")
include("regularizers.jl")

include("utils/general_utils.jl")
include("utils/train_utils.jl")
include("utils/graph_utils.jl")

end # module BTNNs
