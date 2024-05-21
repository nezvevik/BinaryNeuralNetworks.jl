using BTNNs: BTLayer, ternary_quantizer, binary_quantizer, get_binary_regularizer, get_ternary_regularizer
using BTNNs: get_ternary_W, convert2binary_activation, convert2discrete, convert2ternary_weights
using BTNNs: activation_regularizer, weight_regularizer, set_output_regularizer
using Flux: Chain, Dense, logitcrossentropy

ste_model = Chain(
    BTLayer(784, 256,
    binary_quantizer ∘ tanh, ternary_quantizer ∘ tanh),
    BTLayer(256, 10,
    binary_quantizer ∘ tanh, ternary_quantizer ∘ tanh)
)

m = Chain(
    BTLayer(784, 256,
    tanh, tanh, ternary_quantizer, binary_quantizer),
    BTLayer(256, 10,
    tanh, tanh, ternary_quantizer, binary_quantizer),
)

l = BTLayer(784, 10,
tanh, tanh, ternary_quantizer, binary_quantizer)
w = get_W(l)

t = get_ternary_regularizer(1.0)
@time t(w)



using BTNNs: DSTLayer

l = DSTLayer(784, 10)