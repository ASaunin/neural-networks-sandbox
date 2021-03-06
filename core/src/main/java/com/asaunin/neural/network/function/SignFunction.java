package com.asaunin.neural.network.function;

public final class SignFunction implements ActivationFunction {

    @Override
    public double calculate(double value) {
        return value >= 0.0 ? 1.0 : -1.0;
    }

}
