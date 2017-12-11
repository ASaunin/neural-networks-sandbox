package com.asaunin.neural.network.model;

import com.asaunin.neural.network.function.RandomFunction;
import com.asaunin.neural.network.function.SigmoidFunction;

public class Layer {

    private static final SigmoidFunction sigmoidFunction = new SigmoidFunction();

    private final Vector outputs;
    private final Vector inputs;
    private final Vector weights;
    private final Vector dWeights;

    public Layer(int inputSize, int outputSize) {
        this.outputs = new Vector(outputSize);
        this.inputs = new Vector(inputSize + 1);
        this.weights = new Vector((1 + inputSize) * outputSize);
        this.weights.apply(sigmoidFunction);
        this.dWeights = new Vector(weights.size())
                .apply(new RandomFunction());
    }

    public Vector propagate(Vector inputArray) {
        System.arraycopy(inputArray.getValues(), 0, this.inputs.getValues(), 0, inputArray.size());
        inputs.set(inputs.size() - 1, 1); // bias
        int offset = 0;

        for (int i = 0; i < outputs.size(); i++) {
            for (int j = 0; j < inputs.size(); j++) {
                outputs.set(i, outputs.get(i) + weights.get(offset + j) * inputs.get(j));
            }
            outputs.set(i, sigmoidFunction.calculate(outputs.get(i)));
            offset += inputs.size();
        }

        return outputs;
    }

    public Vector train(Vector error, double learningRate, double momentum) {
        int offset = 0;
        final Vector nextError = new Vector(inputs.size());

        for (int i = 0; i < outputs.size(); i++) {

            double delta = error.get(i) * sigmoidFunction.dSigmoid(outputs.get(i)); // cause the output is the sigmoid(x)

            for (int j = 0; j < inputs.size(); j++) {
                final int previousWeightIndex = offset + j;
                final double dw = inputs.get(j) * delta * learningRate;
                nextError.set(j, nextError.get(j) + weights.get(previousWeightIndex) * delta);
                weights.set(previousWeightIndex, weights.get(previousWeightIndex) + dWeights.get(previousWeightIndex) * momentum + dw);
                dWeights.set(previousWeightIndex, dw);
            }

            offset += inputs.size();
        }

        return nextError;
    }

}
