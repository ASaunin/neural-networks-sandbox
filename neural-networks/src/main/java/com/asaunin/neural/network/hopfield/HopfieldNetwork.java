package com.asaunin.neural.network.hopfield;

import com.asaunin.neural.network.function.SignFunction;
import com.asaunin.neural.network.model.Matrix;
import com.asaunin.neural.network.model.Vector;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class HopfieldNetwork {

    private final int size;
    private Matrix matrix;

    public HopfieldNetwork(int size) {
        this.size = size;
        this.matrix = new Matrix(size);
    }

    public HopfieldNetwork train(NormalizedVector vector) {
        final Vector biPattern = vector.validate(size);

        matrix = new Matrix(biPattern)
                .add(matrix);

        log.info("Trained pattern: {}", vector);
        return this;
    }

    public boolean predict(NormalizedVector pattern) {
        final Vector biPattern = pattern.validate(size);

        final Vector result = matrix
                .multiply(biPattern)
                .apply(new SignFunction());

        for (int i = 0; i < size; ++i) {
            if (biPattern.get(i) != result.get(i)) {
                log.info("Failed to recognize pattern: {}", pattern);
                return false;
            }
        }

        log.info("Pattern {} recognized successfully", pattern);
        return true;
    }

}
