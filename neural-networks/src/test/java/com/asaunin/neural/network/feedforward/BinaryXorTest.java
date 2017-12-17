package com.asaunin.neural.network.feedforward;

import com.asaunin.neural.network.model.Vector;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toMap;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

@SuppressWarnings("Duplicates")
@DisplayName("Binary XOR example")
class BinaryXorTest extends BinaryOperationTest {

    private static final HashMap<Vector, Vector> DATA = new LinkedHashMap<Vector, Vector>() {{
        put(Vector.of(0, 0), Vector.of(0));
        put(Vector.of(0, 1), Vector.of(1));
        put(Vector.of(1, 0), Vector.of(1));
        put(Vector.of(1, 1), Vector.of(0));
    }};

    private static final double LEARNING_RATE = 0.3;
    private static final double MOMENTUM = 0.6;
    private static final int ITERATIONS = 500000;
    private static final double ERROR_PROBABILITY = 0.05;

    BinaryXorTest() {
        super(builder -> builder
                .withLearningRate(LEARNING_RATE)
                .withMomentum(MOMENTUM)
                .withMaxIterations(ITERATIONS));
    }

    @BeforeEach
    void train() {
        final Map<Vector, Vector> data = dataProvider()
                .collect(toMap(Map.Entry::getKey, Map.Entry::getValue));

        network.train(data);
    }

    @DisplayName("Network predicts result")
    @ParameterizedTest(name = "Test XOR prediction: {0}")
    @MethodSource("dataProvider")
    void predict(Map.Entry<Vector, Vector> pair) {
        final Vector input = pair.getKey();

        final Vector expectedOutput = pair.getValue();
        final Vector actualOutput = network.predict(input);

        final double actualResult = actualOutput.get(0);
        final double expectedResult = expectedOutput.get(0);

        assertThat(actualResult).isCloseTo(expectedResult, within(ERROR_PROBABILITY));
    }

    private static Stream<Map.Entry<Vector, Vector>> dataProvider() {
        return DATA.entrySet().stream();
    }

}