package com.asaunin.neural.network.model;

import com.asaunin.neural.network.function.ActivationFunction;
import lombok.ToString;

@ToString
public class Vector {

    protected double[] values;

    public Vector(double[] values) {
        this.values = values;
    }

    public Vector(int size) {
        this(new double[size]);
    }

    public static Vector of(int... pattern) {
        final double[] doubles = new double[pattern.length];
        for (int i = 0; i < pattern.length; i++) {
            doubles[i] = pattern[i];
        }
        return new Vector(doubles);
    }

    public Vector validate(int expectedSize) throws IllegalArgumentException {
        if (values.length != expectedSize) {
            throw new IllegalArgumentException(
                    String.format(
                            "Failed to train network: wrong pattern size (actual:%d, expected:%d)",
                            values.length, expectedSize));
        }
        return this;
    }

    public int size() {
        return values.length;
    }

    public double get(int index) {
        return values[index];
    }

    public void set(int index, double value) {
        values[index] = value;
    }

    public <T extends ActivationFunction> Vector apply(T func) {
        values = func.calculate(values);
        return this;
    }

}
