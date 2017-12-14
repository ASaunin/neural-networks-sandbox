package com.asaunin.neural.network.feedforward;

import com.asaunin.neural.network.feedforward.BackPropagationNetwork.BackPropagationNetworkBuilder;

import java.util.function.Consumer;

abstract class BinaryOperationTest {

    private static final int INPUTS_SIZE = 2;
    private static final int OUTPUTS_SIZE = 3;
    private static final int HIDDEN_SIZE = 1;

    final BackPropagationNetwork network;

    BinaryOperationTest(Consumer<BackPropagationNetworkBuilder> network) {
        final BackPropagationNetworkBuilder builder = BackPropagationNetwork.builder();

        network.accept(builder);

        builder.addInputLayer(INPUTS_SIZE)
                .addHiddenLayer(OUTPUTS_SIZE)
                .addOutputLayer(HIDDEN_SIZE);

        this.network = builder.build();
    }

}
