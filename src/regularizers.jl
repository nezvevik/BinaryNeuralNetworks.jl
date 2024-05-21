# binary regularizer
function get_binary_regularizer(pow::Real=1.0)
    # f(x) = min(abs(x + 1), abs(x - 1))^pow
    f(x) = (abs(1f0 - x^2)/(1f0 + abs(x)))^pow
    return x -> (typeof(x) <: Float64 && return f(x)) || (return mean(f.(x)))
end

# ternary regularizer
function get_ternary_regularizer(pow::Real=1.0)
    # f(x) = min(abs(x + 1), abs(x - 1), abs(x))^pow
    f(x) = abs(4*(abs(x) - x*x)/(1f0 + abs(1f0 - 2*abs(x))))^pow
    return x -> (typeof(x) <: Float64 && return f(x)) || (return mean(f.(x)))
end

# Regularizers
function weight_regularizer(Chain)
    sum([weight_regularizer(layer) for layer in Chain.layers])
end

function weight_regularizer(l::BTLayer)
    l.weight_regularizer(get_W(l))
end

# activation regularizer
function activation_regularizer(m::Union{Chain, BTLayer}, x::VecOrMat)
    _, r = m((x, 0.0))
    return r
end

function activation_regularizer(m::Union{Chain, BTLayer}, data::DataLoader)
    r = 0f0
    for (x, _) in data
        r += activation_regularizer(m, x)
    end
    return r / (length(data))
end