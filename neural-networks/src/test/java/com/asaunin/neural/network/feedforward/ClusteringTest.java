package com.asaunin.neural.network.feedforward;

import com.asaunin.neural.network.model.Vector;
import lombok.extern.slf4j.Slf4j;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

@Slf4j
@SuppressWarnings("Duplicates")
public class ClusteringTest {

    private BackPropagationNetwork network;

    private final Vector firstGroup = Vector.of(1, 0, 0);
    private final Vector secondGroup = Vector.of(0, 1, 0);
    private final Vector thirdGroup = Vector.of(0, 0, 1);

    private final Map<Vector, Vector> data = new HashMap<Vector, Vector>() {{
        put(Vector.of(0.1, 0.2), firstGroup);
        put(Vector.of(0.1, 0.2), firstGroup);
        put(Vector.of(0.3, 0.2), firstGroup);
        put(Vector.of(0.15, 0.58), firstGroup);
        put(Vector.of(0.45, 0.7), firstGroup);
        put(Vector.of(0.4, 0.9), firstGroup);

        put(Vector.of(0.4, 1.2), secondGroup);
        put(Vector.of(0.45, 0.95), secondGroup);
        put(Vector.of(0.42, 1), secondGroup);
        put(Vector.of(0.5, 1.1), secondGroup);
        put(Vector.of(0.52, 1.45), secondGroup);

        put(Vector.of(0.6, 0.2), thirdGroup);
        put(Vector.of(0.75, 0.7), thirdGroup);
        put(Vector.of(0.9, 0.34), thirdGroup);
        put(Vector.of(0.85, 0.76), thirdGroup);
        put(Vector.of(0.8, 0.34), thirdGroup);
    }};

    public ClusteringTest() {
        network = BackPropagationNetwork.builder()
                .addInputLayer(2)
                .addHiddenLayer(4)
                .addOutputLayer(3)
                .withLearningRate(0.3)
                .withMomentum(0.6)
                .withMaxIterations(10000)
                .build();
    }

    @Before
    public void train() {
        final List<Vector> inputs = new ArrayList<>(data.keySet());
        final List<Vector> outputs = new ArrayList<>(data.values());

        network.train(inputs, outputs);
    }

    @Test
    public void predict() {
        final Vector input = Vector.of(0.05, 0.12);
        final Vector output = network.predict(input);

        final double firstGroupPrediction = output.get(0);
        final double secondGroupPrediction = output.get(1);
        final double thirdGroupPrediction = output.get(2);
        log.info("For input: {}, predictions by groups are: first={}%, second={}%, third={}%",
                input,
                String.format("%.2f", 100*firstGroupPrediction),
                String.format("%.2f", 100*secondGroupPrediction),
                String.format("%.2f", 100*thirdGroupPrediction));

        final double accuracy = 0.05;
        assertThat(firstGroupPrediction).isCloseTo(1, within(accuracy));
        assertThat(secondGroupPrediction).isCloseTo(0, within(accuracy));
        assertThat(thirdGroupPrediction).isCloseTo(0, within(accuracy));

    }

}
