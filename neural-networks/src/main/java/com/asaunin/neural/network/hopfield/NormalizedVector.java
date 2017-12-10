package com.asaunin.neural.network.hopfield;

import com.asaunin.neural.network.model.Vector;
import lombok.ToString;

@ToString
public class NormalizedVector extends Vector {

    public NormalizedVector(double[] values) {
        super(values);
        normalize();
    }

    private void normalize() {
        for (int i = 0; i < values.length; i++) {
            values[i] = (values[i] <= 0.0) ? -1.0 : 1.0;
        }
    }

    public static NormalizedVector of(int... pattern) {
        final double[] doubles = new double[pattern.length];
        for (int i = 0; i < pattern.length; i++) {
            doubles[i] = pattern[i];
        }
        return new NormalizedVector(doubles);

    }

}
