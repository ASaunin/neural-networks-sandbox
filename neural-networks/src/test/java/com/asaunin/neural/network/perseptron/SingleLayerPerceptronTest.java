package com.asaunin.neural.network.perseptron;

import com.asaunin.neural.network.model.Matrix;
import com.asaunin.neural.network.model.Vector;
import org.junit.Ignore;
import org.junit.Test;

import static org.assertj.core.api.AssertionsForInterfaceTypes.assertThat;

public class SingleLayerPerceptronTest {

    @Test
    public void networkIsTrainableForAndFunction() {
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

    @Test(timeout = 1000)
    @Ignore("Here comes Single layer perceptron problem, cause for predict operation hidden layer is necessary")
    public void networkIsNotTrainableForXorFunction() {
        final double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        final Matrix matrix = new Matrix(inputs);
        final Vector xor = Vector.of(0, 1, 1, 0);

        final SingleLayerPerceptron network = new SingleLayerPerceptron(matrix, xor);
        network.train(0.1);

        assertThat(network.predict(Vector.of(0, 0))).isEqualTo(0.0);
        assertThat(network.predict(Vector.of(0, 1))).isEqualTo(1.0);
        assertThat(network.predict(Vector.of(1, 0))).isEqualTo(1.0);
        assertThat(network.predict(Vector.of(1, 1))).isEqualTo(0.0);
    }

}