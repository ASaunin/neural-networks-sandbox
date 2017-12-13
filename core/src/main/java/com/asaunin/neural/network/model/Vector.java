package com.asaunin.neural.network.model;

import com.asaunin.neural.network.function.ActivationFunction;
import lombok.Getter;

import java.util.Arrays;
import java.util.StringJoiner;
import java.util.function.Function;

@Getter
public class Vector implements Function<ActivationFunction, Vector> {

    private final double[] values;

    public Vector(double[] values) {
        this.values = Arrays.copyOf(values, values.length);
    }

    public Vector(int size) {
        this.values = new double[size];
    }

    public static Vector of(int... pattern) {
        final double[] doubles = new double[pattern.length];
        for (int i = 0; i < pattern.length; i++) {
            doubles[i] = pattern[i];
        }
        return new Vector(doubles);
    }

    public static Vector of(double... pattern) {
        final double[] doubles = new double[pattern.length];
        System.arraycopy(pattern, 0, doubles, 0, pattern.length);
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

    @Override
    public Vector apply(ActivationFunction func) {
        func.calculate(values);
        return this;
    }

    @Override
    public String toString() {
        final StringJoiner joiner = new StringJoiner(", ", "[", "]");
        for (double value : values) {
            joiner.add(String.valueOf(value));
        }
        return joiner.toString();
    }

}
