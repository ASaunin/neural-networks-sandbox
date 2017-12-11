package com.asaunin.neural.network.feedforward;

import com.asaunin.neural.network.model.Vector;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

// TODO: 12.12.2017 Better to switch to another test framework that fully supports @Parametrized
@SuppressWarnings("Duplicates")
public class AndFunctionTest extends BackPropagationNetworkTest {

    private final List<Vector> inputs = Arrays.asList(
            Vector.of(0, 0),
            Vector.of(0, 1),
            Vector.of(1, 0),
            Vector.of(1, 1));

    private final List<Vector> outputs = Arrays.asList(
            Vector.of(0),
            Vector.of(0),
            Vector.of(0),
            Vector.of(1));

    @Before
    @Override
    public void train() {
        network.train(inputs, outputs);
    }

    @Test
    @Override
    public void predict() {
        for (int index = 0; index < inputs.size(); index++) {
            final Vector input = inputs.get(index);
            final Vector output = network.predict(input);
            final double actualResult = output.get(0);

            final double expectedResult = outputs.get(index).get(0);

            assertThat(actualResult).isCloseTo(expectedResult, within(getAccuracy()));
        }
    }

}