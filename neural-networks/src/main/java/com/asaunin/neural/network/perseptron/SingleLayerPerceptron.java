package com.asaunin.neural.network.perseptron;

import com.asaunin.neural.network.function.StepFunction;
import com.asaunin.neural.network.model.Matrix;
import com.asaunin.neural.network.model.Vector;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@RequiredArgsConstructor
public class SingleLayerPerceptron {

    private final Matrix inputs;
    private final Vector outputs;
    private final Vector weights;

    public SingleLayerPerceptron(Matrix inputs, Vector outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
        this.weights = new Vector(inputs.columns());
    }

    public void train(double learningRate) {
        double totalError = 1;
        int iteration = 0;

        while (totalError != 0) {
            totalError = 0;

            for (int i = 0; i < outputs.size(); i++) {
                final Vector input = inputs.get(i);
                final double output = predict(input);
                final double error = outputs.get(i) - output;

                totalError += error;

                for (int j = 0; j < weights.size(); j++) {
                    final double inputValue = inputs.get(i, j);
                    final double oldWeight = weights.get(j);
                    final double newWeight = oldWeight + learningRate * inputValue * error;
                    weights.set(j, newWeight);
                    log.debug("Weight {} updated from: {} to: {}", j, oldWeight, newWeight);
                }
            }

            log.info("Next training iteration: {}, error: {}", iteration++, totalError);
        }

        log.info("Perceptron trained in: {} iterations", iteration);
    }

    public double predict(Vector input) {
        double sum = 0f;
        for (int i = 0; i < input.size(); ++i)
            sum = sum + weights.get(i) * input.get(i);

        return new StepFunction().calculate(sum);
    }

}
