"""
The `FeatureQuantizer` module defines a struct and associated functions for feature quantization within a neural network. 
Feature quantization involves discretizing input features using learnable weights and biases, and applying an output quantization function.

```julia
function (q::FeatureQuantizer)(x)
```

`FeatureQuantizer` serves as a functor and it applies the feature quantization operation to the input data x 
using the weights and biases stored in the FeatureQuantizer object and returns the quantized output.
"""
struct FeatureQuantizer{M<:AbstractMatrix, B, Q}
    weight::M
    bias::B
    output_quantizer::Q

    function FeatureQuantizer(
        weight::W,
        bias::B;
        output_quantizer::Q = binary_quantizer,
    ) where {W<:AbstractMatrix, B<:AbstractMatrix, Q}

        return new{W, B, Q}(weight, bias, output_quantizer)
    end
end

Flux.@functor FeatureQuantizer

function FeatureQuantizer(
    dim::Int64,
    k::Int64;
    init_weight = glorot_uniform,
    init_bias = (d...) -> randn(Float32, d...),
    kwargs...
)
    return FeatureQuantizer(init_weight(dim, k), init_bias(dim, k); kwargs...)
end

function Base.show(io::IO, q::FeatureQuantizer)
    print(io, "FeatureQuantizer(")
    print(io, size(q.weight, 1), " => ", prod(size(q.weight)))
    print(io, "; quantizer=$(q.output_quantizer))")
end

function (q::FeatureQuantizer)(x)
    y = _forward_pass(q.weight, q.bias, x)
    return q.output_quantizer.(y)
end


function _forward_pass(w, b, x::AbstractVector)
    return vec(_forward_pass(w, b, reshape(x, length(x), 1)))
end

"""
This is an internal function and performs the forward pass of the feature quantization layer for input data with multiple dimensions.
It computes the weighted sum of input data, applies biases, and quantizes the result. 
There is also a version for supplying one-dimensional input vector.
"""
function _forward_pass(w, b, x)
    w1, b1, x1 = size(w, 1), size(b, 1), size(x, 1)
    if !(w1 == b1 == x1)
        msg = "first dimension of weight ($w1), bias ($b1) and x ($x1) must match"
        throw(DimensionMismatch(msg))
    end
    y = similar(x, length(w), size(x, 2))
    for col in axes(x, 2)
        for j in axes(w,2), i in axes(x,1)
            idx = (i-1)*size(w,2) + j
            y[idx,col] = x[i,col] * w[i, j] + b[i, j]
        end
    end
    return y
end

"""
This function defines the reverse-mode automatic differentiation (AD) rule for the `_forward_pass` function.
It specifies how gradients are propagated backward through the quantization layer during backpropagation.
    
- `project_w`: A function to project the gradient with respect to weights.
- `project_b`: A function to project the gradient with respect to biases.
- `project_x`: A function to project the gradient with respect to input data.

This function returns a tuple containing gradients with respect to weights, biases, and input data, along with the projected gradients for each.

This internal function is used for reverse-mode AD during backpropagation. 
It computes gradients for the feature quantization layer and returns the gradients for weights, biases, and input data.
Δy is the gradient with respect to the output.
"""
function ChainRulesCore.rrule(::typeof(_forward_pass), w, b, x)
    project_w = ProjectTo(w)
    project_b = ProjectTo(b)
    project_x = ProjectTo(x)

    function FeatureQuantizer_pullback(Δy)
        Δw, Δb, Δx = zero.((w, b, x))

        for col in axes(x, 2)
            for j in axes(w,2), i in axes(x,1)
                if !ismissing(x[i, col])
                    Δw[i, j] += x[i,col] * Δy[i, col]
                    Δb[i, j] += Δy[i, col]
                    Δx[i,col] +=  w[i, j] * Δy[i, col]
                end
            end
        end
        return NoTangent(), project_w(Δw), project_b(Δb), project_x(Δx)
    end
    return _forward_pass(w, b, x), FeatureQuantizer_pullback
end


