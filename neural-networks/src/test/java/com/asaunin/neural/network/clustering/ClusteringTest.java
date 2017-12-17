package com.asaunin.neural.network.clustering;

import com.asaunin.neural.network.feedforward.BackPropagationNetwork;
import com.asaunin.neural.network.model.Vector;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

@Slf4j
@DisplayName("Clustering example")
class ClusteringTest {

    private static final Vector FIRST_GROUP = Vector.of(1, 0, 0);
    private static final Vector SECOND_GROUP = Vector.of(0, 1, 0);
    private static final Vector THIRD_GROUP = Vector.of(0, 0, 1);

    private final Map<Vector, Vector> data = new LinkedHashMap<Vector, Vector>() {{
        put(Vector.of(0.1, 0.2), FIRST_GROUP);
        put(Vector.of(0.1, 0.2), FIRST_GROUP);
        put(Vector.of(0.3, 0.2), FIRST_GROUP);
        put(Vector.of(0.15, 0.58), FIRST_GROUP);
        put(Vector.of(0.45, 0.7), FIRST_GROUP);
        put(Vector.of(0.4, 0.9), FIRST_GROUP);

        put(Vector.of(0.4, 1.2), SECOND_GROUP);
        put(Vector.of(0.45, 0.95), SECOND_GROUP);
        put(Vector.of(0.42, 1), SECOND_GROUP);
        put(Vector.of(0.5, 1.1), SECOND_GROUP);
        put(Vector.of(0.52, 1.45), SECOND_GROUP);

        put(Vector.of(0.6, 0.2), THIRD_GROUP);
        put(Vector.of(0.75, 0.7), THIRD_GROUP);
        put(Vector.of(0.9, 0.34), THIRD_GROUP);
        put(Vector.of(0.85, 0.76), THIRD_GROUP);
        put(Vector.of(0.8, 0.34), THIRD_GROUP);
    }};

    private final BackPropagationNetwork network;

    ClusteringTest() {
        network = BackPropagationNetwork.builder()
                .addInputLayer(2)
                .addHiddenLayer(4)
                .addOutputLayer(3)
                .withLearningRate(0.3)
                .withMomentum(0.6)
                .withMaxIterations(10000)
                .build();
    }

    @BeforeEach
    void train() {
        network.train(data);
    }

    @DisplayName("When point is close to the group, than predict that group")
    @Test
    void predict() {
        final Vector input = Vector.of(0.05, 0.12);
        final Vector output = network.predict(input);

        final double firstGroupPrediction = output.get(0);
        final double secondGroupPrediction = output.get(1);
        final double thirdGroupPrediction = output.get(2);
        log.info("For input: {}, predictions by groups are: first={}%, second={}%, third={}%",
                input,
                String.format("%.2f", 100 * firstGroupPrediction),
                String.format("%.2f", 100 * secondGroupPrediction),
                String.format("%.2f", 100 * thirdGroupPrediction));

        final double error_probability = 0.05;
        assertThat(firstGroupPrediction).isCloseTo(1, within(error_probability));
        assertThat(secondGroupPrediction).isCloseTo(0, within(error_probability));
        assertThat(thirdGroupPrediction).isCloseTo(0, within(error_probability));
    }

}
