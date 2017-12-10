package com.asaunin.neural.network.function;

@FunctionalInterface
public interface ActivationFunction {

    double calculate(double value);

    default double[] calculate(double[] values) {
        double[] output = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            output[i] = calculate(values[i]);
        }
        return output;
    }

}
