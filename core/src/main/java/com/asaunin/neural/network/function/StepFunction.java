package com.asaunin.neural.network.function;

public final class StepFunction implements ActivationFunction {

    @Override
    public double calculate(double value) {
        return value >= 1.0 ? 1.0 : 0.0;
    }

}
