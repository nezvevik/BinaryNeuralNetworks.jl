function convert2discrete(m::Chain)
    Chain(map(convert2discrete, m.layers))
end

# function convert2discrete(m::Cha2



function convert2discrete(l::BTLayer)
    BTLayer(get_ternary_W(l), copy(l.b), l.output_quantizer ∘ l.σ, identity, identity, identity, l.weight_regularizer, l.output_regularizer, l.batchnorm)
end
# function convert2discrete(l::BTLayer, tq)
#     BTLayer(tq(l.weight_compressor.(l.θ)), copy(l.b), l.output_quantizer ∘ l.σ, identity, identity, identity, l.weight_regularizer, l.output_regularizer, l.batchnorm)
# end


function convert2binary_activation(m::Chain)
    Chain(map(convert2binary_activation, m.layers))
end

# ternary weights
function convert2ternary_weights(m::Chain)
    Chain(map(convert2ternary_weights, m.layers))
end

function convert2ternary_weights(l::BTLayer)
    BTLayer(get_ternary_W(l), copy(l.b), l.σ, identity, identity, l.output_quantizer, l.weight_regularizer, l.output_regularizer, l.batchnorm)
end

function convert2binary_activation(l::BTLayer)
    BTLayer(l.θ, copy(l.b), l.output_quantizer ∘ l.σ, l.weight_compressor, identity, identity, l.weight_regularizer, l.output_regularizer, l.batchnorm)
end



