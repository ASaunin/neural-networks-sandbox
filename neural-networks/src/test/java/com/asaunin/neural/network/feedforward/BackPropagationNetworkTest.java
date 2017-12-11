package com.asaunin.neural.network.feedforward;

import lombok.Getter;
import lombok.Setter;
import org.junit.Before;
import org.junit.Test;

public abstract class BackPropagationNetworkTest {

    private static final int INPUTS_SIZE = 2;
    private static final int OUTPUTS_SIZE = 3;
    private static final int HIDDEN_SIZE = 1;

    @Getter @Setter private double learningRate = 0.3;
    @Getter @Setter private double momentum = 0.6;
    @Getter @Setter private int iterations = 10000;
    @Getter @Setter private double accuracy = 0.05;

    final BackPropagationNetwork network = BackPropagationNetwork.builder()
            .addInputLayer(INPUTS_SIZE)
            .addHiddenLayer(OUTPUTS_SIZE)
            .addOutputLayer(HIDDEN_SIZE)
            .withLearningRate(learningRate)
            .withMomentum(momentum)
            .withMaxIterations(iterations)
            .build();

    @Before
    abstract void train();

    @Test
    abstract void predict();

}
