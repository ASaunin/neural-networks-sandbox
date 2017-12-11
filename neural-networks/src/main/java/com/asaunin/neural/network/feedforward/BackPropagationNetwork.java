package com.asaunin.neural.network.feedforward;

import com.asaunin.neural.network.model.Layer;
import com.asaunin.neural.network.model.Vector;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Slf4j
public class BackPropagationNetwork {

    private final double learningRate;
    private final double momentum;
    private final int maxIterations;
    private final List<Layer> layers;

    public BackPropagationNetwork(List<Layer> layers, double learningRate, double momentum, int maxIterations) {
        this.layers = Collections.unmodifiableList(layers);
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.maxIterations = maxIterations;
    }

    public static BackPropagationNetworkBuilder builder() {
        return new BackPropagationNetworkBuilder();
    }

    public Vector predict(Vector input) {
        Vector inputActivation = input;
        for (Layer layer : layers) {
            inputActivation = layer.propagate(inputActivation);
        }
        return inputActivation;
    }

    public void train(List<Vector> inputs, List<Vector> outputs) {
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            for (int i = 0; i < outputs.size(); i++) {
                train(inputs.get(i), outputs.get(i));
            }

            for (int i = 0; i < outputs.size(); i++) {
                final Vector input = inputs.get(i);
                final Vector output = predict(input);
                if (iteration % (maxIterations / 100) == 0) {
                    log.info("Train iteration: {}, Inputs: {}, Outputs: {}", iteration, input, output);
                }
            }
        }

    }

    private void train(Vector input, Vector output) {
        final Vector calculatedOutput = predict(input);
        Vector error = new Vector(calculatedOutput.size());

        for (int i = 0; i < error.size(); i++) {
            error.set(i, output.get(i) - calculatedOutput.get(i));
        }

        for (int i = layers.size() - 1; i >= 0; i--) {
            error = layers.get(i).train(error, learningRate, momentum);
        }
    }

    public static class BackPropagationNetworkBuilder {

        private int inputSize;
        private int outputSize;
        private List<Integer> hiddenSizes = new ArrayList<>();
        private double learningRate;
        private double momentum;
        private int maxIterations;

        public BackPropagationNetworkBuilder addInputLayer(int size) {
            this.inputSize = size;
            return this;
        }

        public BackPropagationNetworkBuilder addHiddenLayer(int size) {
            this.hiddenSizes.add(size);
            return this;
        }

        public BackPropagationNetworkBuilder addOutputLayer(int size) {
            this.outputSize = size;
            return this;
        }

        public BackPropagationNetworkBuilder withLearningRate(double rate) {
            this.learningRate = rate;
            return this;
        }

        public BackPropagationNetworkBuilder withMomentum(double momentum) {
            this.momentum = momentum;
            return this;
        }

        public BackPropagationNetworkBuilder withMaxIterations(int iterations) {
            this.maxIterations = iterations;
            return this;
        }

        public BackPropagationNetwork build() {
            final List<Layer> layers = new ArrayList<>();
            if (hiddenSizes.size() == 0) {
                layers.add(new Layer(inputSize, outputSize));
            } else {
                layers.add(new Layer(inputSize, hiddenSizes.get(0)));
                for (int i = 0; i < hiddenSizes.size() - 1; i++) {
                    layers.add(new Layer(hiddenSizes.get(i), hiddenSizes.get(i + 1)));
                }
                layers.add(new Layer(hiddenSizes.get(hiddenSizes.size() - 1), outputSize));
            }
            return new BackPropagationNetwork(layers, learningRate, momentum, maxIterations);
        }
    }

}
