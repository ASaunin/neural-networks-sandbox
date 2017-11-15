package com.asaunin.neural.network.hopfield;

import lombok.ToString;

@ToString
public class Matrix {

    private final double[][] values;

    public Matrix(int size) {
        this.values = new double[size][size];
    }

    public Matrix(double[][] values) {
        this.values = values;
    }

    public Matrix(Vector vector) {
        final int size = vector.size();
        values = new double[size][size];

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                if (i != j) {
                    values[i][j] = vector.get(i) * vector.get(j);
                }
            }
        }
    }

    public Vector multiply(Vector vector) {
        final Vector result = new Vector(this.values.length);

        for (int i = 0; i < this.values.length; ++i) {
            double sum = 0;
            for (int j = 0; j < this.values.length; ++j) {
                sum += this.values[i][j] * vector.get(j);
            }
            result.set(i, sum);
        }

        return result;
    }

    public Matrix add(Matrix matrix) {
        for (int i = 0; i < values.length; ++i) {
            for (int j = 0; j < values.length; ++j) {
                values[i][j] = values[i][j] + matrix.values[i][j];
            }
        }

        return this;
    }
}
