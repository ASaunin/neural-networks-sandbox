package com.asaunin.neural.network.ocr;

import com.asaunin.neural.network.feedforward.BackPropagationNetwork;
import com.asaunin.neural.network.model.Vector;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.IntStream;

import static org.assertj.core.api.Assertions.within;
import static org.assertj.core.api.AssertionsForClassTypes.assertThat;

@Slf4j
@DisplayName("Digit OCR example")
class DigitRecognitionTest {

    private static final int INPUTS_SIZE = 64;
    private static final int OUTPUTS_SIZE = 10;
    private static final int HIDDEN_SIZE = 15;
    private static final double LEARNING_RATE = 0.3;
    private static final double MOMENTUM = 0.6;
    private static final int ITERATIONS = 500000;
    private static final double ERROR_PROBABILITY = 0.25;

    private final BackPropagationNetwork network;
    private final ImageReader reader = new ImageReader();

    DigitRecognitionTest() {
        this.network = BackPropagationNetwork.builder()
                .addInputLayer(INPUTS_SIZE)
                .addHiddenLayer(HIDDEN_SIZE)
                .addOutputLayer(OUTPUTS_SIZE)
                .withLearningRate(LEARNING_RATE)
                .withMomentum(MOMENTUM)
                .withMaxIterations(ITERATIONS)
                .build();
    }

    @BeforeEach
    void train() throws Exception {
        final Map<Vector, Vector> data = new LinkedHashMap<>();
        for (int index = 0; index < OUTPUTS_SIZE; index++) {
            final File file = new File(getClass().getClassLoader().getResource(String.format("ocr/train/%d.jpg", index)).toURI());
            final Vector input = reader.read(file).toVector();
            final Vector output = getOutputVector(index, OUTPUTS_SIZE);
            data.put(input, output);
        }

        network.train(data);
    }

    @DisplayName("Network predicts digit of 5")
    @Test
    void predict() throws Exception {
        final int expectedResult = 5;
        final File file = new File(getClass().getClassLoader().getResource(String.format("ocr/train/%d.jpg", expectedResult)).toURI());
        final Vector input = reader.read(file).toVector();
        final Vector output = network.predict(input);

        for (int index = 0; index < OUTPUTS_SIZE; index++) {
            final double actualResult = output.get(index);
            log.info("5 is similar to: {} with probability of: {}%", index, String.format("%.2f", 100 * actualResult));
            if (index == 5) {
                assertThat(actualResult)
                        .as(String.format("%d prediction", index))
                        .isCloseTo(1, within(ERROR_PROBABILITY));
            } else {
                assertThat(actualResult)
                        .as(String.format("%d prediction", index))
                        .isCloseTo(0, within(ERROR_PROBABILITY));
            }
        }
    }

    private Vector getOutputVector(int index, int size) {
        return Vector.of(IntStream.range(0, size)
                .map(i -> (index == i) ? 1 : 0)
                .toArray());
    }

}
