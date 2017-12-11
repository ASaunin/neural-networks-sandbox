package com.asaunin.neural.network.function;

import java.util.Random;

public final class RandomFunction implements ActivationFunction {

    @Override
    public double calculate(double value) {
        final Random random = new Random();
        return (random.nextDouble() - 0.5) * 4; //[-2, +2]
    }

}
