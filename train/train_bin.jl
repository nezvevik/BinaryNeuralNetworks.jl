using BinaryNeuralNetwork: BinaryLayer, convert2normal, ternary_quantizer
using Flux: relu, Chain, AdaBelief


include("utils.jl")

train_data, test_data = createloader(batchsize=256)

# .97 on train_data with relu
# .83 on train_data with sign
model = Chain(
    BinaryLayer(28^2, 256, sign),
    BinaryLayer(256, 10, identity)
)

# accuracy before traini1ng
accuracy(train_data, model)
accuracy(test_data, model)


# train model
history = train_model(model, AdaBelief(), train_data, test_data; epochs=15)


# accuracy after training
accuracy(train_data, model)
accuracy(test_data, model)

# convert to a normal layer with binary weights 
normal_model = convert2normal(model)

# check accuracy after conversion, should be equal to the accuracy of the binary model
accuracy(train_data, normal_model)
accuracy(test_data, normal_model)
 