package com.asaunin.neural.network.perseptron;

import com.asaunin.neural.network.model.Matrix;
import com.asaunin.neural.network.model.Vector;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.time.Duration;

import static org.assertj.core.api.AssertionsForInterfaceTypes.assertThat;
import static org.junit.jupiter.api.Assertions.assertTimeout;

@DisplayName("Single layer perceptron example")
class SingleLayerPerceptronTest {

    @DisplayName("Binary AND prediction")
    @Test
    void networkIsTrainableForAndFunction() {
        final double[][] inputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
        final Matrix matrix = new Matrix(inputs);
        final Vector and = Vector.of(0, 0, 0, 1);

        final SingleLayerPerceptron network = new SingleLayerPerceptron(matrix, and);
        network.train(0.1);

        assertThat(network.predict(Vector.of(0, 0))).isEqualTo(0.0);
        assertThat(network.predict(Vector.of(0, 1))).isEqualTo(0.0);
        assertThat(network.predict(Vector.of(1, 0))).isEqualTo(0.0);
        assertThat(network.predict(Vector.of(1, 1))).isEqualTo(1.0);
    }

    @DisplayName("Binary XOR prediction")
    @Test
    @Disabled("Here comes Single layer perceptron problem, cause for predict operation hidden layer is necessary")
    void networkIsNotTrainableForXorFunction() {
        final double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        final Matrix matrix = new Matrix(inputs);
        final Vector xor = Vector.of(0, 1, 1, 0);

        final SingleLayerPerceptron network = new SingleLayerPerceptron(matrix, xor);
        assertTimeout(Duration.ofSeconds(10), () -> network.train(0.1));

        assertThat(network.predict(Vector.of(0, 0))).isEqualTo(0.0);
        assertThat(network.predict(Vector.of(0, 1))).isEqualTo(1.0);
        assertThat(network.predict(Vector.of(1, 0))).isEqualTo(1.0);
        assertThat(network.predict(Vector.of(1, 1))).isEqualTo(0.0);
    }

}