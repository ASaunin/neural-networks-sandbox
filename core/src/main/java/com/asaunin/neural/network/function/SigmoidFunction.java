package com.asaunin.neural.network.function;

public final class SigmoidFunction implements ActivationFunction {

    @Override
    public double calculate(double value) {
        return (1 / (1 + Math.exp(-value)));
    }

    public double dSigmoid(double value) {
        return value * (1 - value);
    }

}
