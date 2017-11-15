package com.asaunin.neural.network.hopfield;

import com.asaunin.neural.network.function.StepFunction;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class HopfieldNetwork {

    private final int size;
    private Matrix matrix;

    public HopfieldNetwork(int size) {
        this.size = size;
        this.matrix = new Matrix(size);
    }

    public HopfieldNetwork train(Vector vector) {
        final Vector biPattern = vector.validate(size);

        matrix = new Matrix(biPattern)
                .add(matrix);

        log.info("Trained pattern: {}", vector);
        return this;
    }

    public boolean validate(Vector pattern) {
        final Vector biPattern = pattern.validate(size);

        final Vector result = matrix
                .multiply(biPattern)
                .apply(new StepFunction());

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
