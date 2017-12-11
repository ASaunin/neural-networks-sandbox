package com.asaunin.neural.network.function;

@FunctionalInterface
public interface ActivationFunction {

    double calculate(double value);

    default void calculate(double[] values) {
        for (int i = 0; i < values.length; i++) {
            values[i] = calculate(values[i]);
        }
    }

}
