package com.asaunin.neural.network.function;

public class StepFunction implements ActivationFunction {

    @Override
    public double calculate(double value) {
        return value >= 0.0 ? 1.0 : -1.0;
    }

}
