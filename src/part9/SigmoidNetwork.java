package part9;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.jblas.DoubleMatrix;
import org.jblas.util.Random;

public class SigmoidNetwork implements Serializable {

	private static final long serialVersionUID = 1L;

	/**
	 * Rate of precision that indicates when the net should stop learning. Possible
	 * values are in range 0..1
	 */
	private static final double PARAM_PRECISION_RATE = 1;

	private int numLayers;

	private DoubleMatrix[] weights;
	private DoubleMatrix[] biases;

	public SigmoidNetwork(int... sizes) {
		this.numLayers = sizes.length;

		this.biases = new DoubleMatrix[sizes.length - 1];
		this.weights = new DoubleMatrix[sizes.length - 1];

		// Storing biases
		for (int i = 1; i < sizes.length; i++) {
			double[][] temp = new double[sizes[i]][];
			for (int j = 0; j < sizes[i]; j++) {
				double[] b = new double[] { Random.nextGaussian() };
				temp[j] = b;
			}
			biases[i - 1] = new DoubleMatrix(temp);
		}
		// Storing weights
		for (int i = 1; i < sizes.length; i++) {
			double[][] temp = new double[sizes[i]][];
			for (int j = 0; j < sizes[i]; j++) {
				double[] w = new double[sizes[i - 1]];
				for (int k = 0; k < sizes[i - 1]; k++) {
					w[k] = Random.nextGaussian();
				}
				temp[j] = w;
			}
			weights[i - 1] = new DoubleMatrix(temp);
		}
	}

	/**
	 * @param trainingData  - list of arrays (x, y) representing the training inputs
	 *                      and corresponding desired outputs
	 * @param epochs        - the number of epochs to train for
	 * @param miniBatchSize - the size of the mini-batches to use when sampling
	 * @param eta           - the learning rate, η
	 * @param testData      - the test data use to evaluate the net
	 * @return void
	 */
	public void SGD(List<double[][]> trainingData, int epochs, int miniBatchSize, double eta,
			List<double[][]> testData) {

		int nTest = 0;

		int n = trainingData.size();

		if (testData != null) {
			nTest = testData.size();
		}

		for (int j = 0; j < epochs; j++) {
			Collections.shuffle(trainingData);
			List<List<double[][]>> miniBatches = new ArrayList<>();
			for (int k = 0; k < n; k += miniBatchSize) {
				miniBatches.add(trainingData.subList(k, k + miniBatchSize));
			}
			for (List<double[][]> miniBatch : miniBatches) {
				updateMiniBatch(miniBatch, eta);
			}

			if (testData != null) {
				int e = evaluate(testData);
				System.out.println(String.format("Epoch %d: %d / %d", j, e, nTest));
				if (e >= nTest * PARAM_PRECISION_RATE) {
					try {
						Util.serialize(this);
					} catch (IOException e1) {
						e1.printStackTrace();
					}
					break;
				}
			} else {
				System.out.println(String.format("Epoch %d complete", j));
			}
		}

	}

	/**
	 * Update the network’s weights and biases by applying gradient descent using
	 * backpropagation to a single mini batch. The "mini_batch" is a list of arrays
	 * "(x, y)", and "eta" is the learning rate.
	 * 
	 * @param miniBatch - part of a training data
	 * @param eta       - the learning rate
	 */
	private void updateMiniBatch(List<double[][]> miniBatch, double eta) {
		DoubleMatrix[] nablaB = new DoubleMatrix[biases.length];
		DoubleMatrix[] nablaW = new DoubleMatrix[weights.length];

		for (int i = 0; i < nablaB.length; i++) {
			nablaB[i] = new DoubleMatrix(biases[i].getRows(), biases[i].getColumns());
		}
		for (int i = 0; i < nablaW.length; i++) {
			nablaW[i] = new DoubleMatrix(weights[i].getRows(), weights[i].getColumns());
		}

		for (double[][] inputOutput : miniBatch) {
			DoubleMatrix[][] deltas = backProp(inputOutput);

			DoubleMatrix[] deltaNablaB = deltas[0];
			DoubleMatrix[] deltaNablaW = deltas[1];

			for (int i = 0; i < nablaB.length; i++) {
				nablaB[i] = nablaB[i].add(deltaNablaB[i]);
			}
			for (int i = 0; i < nablaW.length; i++) {
				nablaW[i] = nablaW[i].add(deltaNablaW[i]);
			}
		}
		for (int i = 0; i < biases.length; i++) {
			biases[i] = biases[i].sub(nablaB[i].mul(eta / miniBatch.size()));
		}
		for (int i = 0; i < weights.length; i++) {
			weights[i] = weights[i].sub(nablaW[i].mul(eta / miniBatch.size()));
		}
	}

	/**
	 * Return an array (nablaB , nablaW) representing the gradient for the cost
	 * function C. "nablaB" and "nablaW" are layer-by-layer arrays of DoubleMatrices
	 * , similar to this.biases and this.weights.
	 * 
	 * @param inputsOutputs
	 * @return
	 */
	private DoubleMatrix[][] backProp(double[][] inputsOuputs) {
		DoubleMatrix[] nablaB = new DoubleMatrix[biases.length];
		DoubleMatrix[] nablaW = new DoubleMatrix[weights.length];

		for (int i = 0; i < nablaB.length; i++) {
			nablaB[i] = new DoubleMatrix(biases[i].getRows(), biases[i].getColumns());
		}
		for (int i = 0; i < nablaW.length; i++) {
			nablaW[i] = new DoubleMatrix(weights[i].getRows(), weights[i].getColumns());
		}

		// FeedForward
		DoubleMatrix activation = new DoubleMatrix(inputsOuputs[0]);
		DoubleMatrix[] activations = new DoubleMatrix[numLayers];
		activations[0] = activation;
		DoubleMatrix[] zs = new DoubleMatrix[numLayers - 1];

		for (int i = 0; i < numLayers - 1; i++) {
			double[] scalars = new double[weights[i].rows];
			for (int j = 0; j < weights[i].rows; j++) {
				scalars[j] = weights[i].getRow(j).dot(activation) + biases[i].get(j);
			}
			DoubleMatrix z = new DoubleMatrix(scalars);
			zs[i] = z;
			activation = sigmoid(z);
			activations[i + 1] = activation;
		}

		// Backward pass
		DoubleMatrix output = new DoubleMatrix(inputsOuputs[1]);
		DoubleMatrix delta = costDerivative(activations[activations.length - 1], output)
				.mul(sigmoidPrime(zs[zs.length - 1])); // BP1
		nablaB[nablaB.length - 1] = delta; // BP3
		nablaW[nablaW.length - 1] = delta.mmul(activations[activations.length - 2].transpose()); // BP4
		for (int layer = 2; layer < numLayers; layer++) {
			DoubleMatrix z = zs[zs.length - layer];
			DoubleMatrix sp = sigmoidPrime(z);
			delta = weights[weights.length + 1 - layer].transpose().mmul(delta).mul(sp); // BP2
			nablaB[nablaB.length - layer] = delta; // BP3
			nablaW[nablaW.length - layer] = delta.mmul(activations[activations.length - 1 - layer].transpose()); // BP4
		}
		return new DoubleMatrix[][] { nablaB, nablaW };
	}

	/**
	 * 
	 * @param a - activation vector - the 1st layer also called the input layer
	 * @return DoubleMatrix - vector containing output from the network consisting
	 *         of float numbers between 0 and 1
	 */
	public DoubleMatrix feedForward(DoubleMatrix a) {
		for (int i = 0; i < numLayers - 1; i++) {
			double[] z = new double[weights[i].rows];
			for (int j = 0; j < weights[i].rows; j++) {
				z[j] = weights[i].getRow(j).dot(a) + biases[i].get(j);
			}
			DoubleMatrix output = new DoubleMatrix(z);
			a = sigmoid(output);
		}
		return a;
	}

	/**
	 * 
	 * @param z - input vector created by finding dot product of weights and inputs
	 *          and added a bias of a neuron
	 * @return output vector - inputs for the next layer
	 */
	private DoubleMatrix sigmoid(DoubleMatrix z) {
		double[] output = new double[z.length];
		for (int i = 0; i < output.length; i++) {
			output[i] = 1 / (1 + Math.exp(-z.get(i)));
		}
		return new DoubleMatrix(output);
	}

	/**
	 * @param testData - the test data used to evaluate the net
	 * @return the number of test inputs for which the neural network outputs the
	 *         correct result
	 */
	private int evaluate(List<double[][]> testData) {
		int sum = 0;
		for (double[][] inputOutput : testData) {
			DoubleMatrix x = new DoubleMatrix(inputOutput[0]);
			DoubleMatrix y = new DoubleMatrix(inputOutput[1]);
			DoubleMatrix netOutput = feedForward(x);
			if (netOutput.argmax() == y.argmax()) {
				sum++;
			}
		}
		return sum;
	}

	private DoubleMatrix sigmoidPrime(DoubleMatrix z) {
		return sigmoid(z).mul(sigmoid(z).rsub(1));
	}

	private DoubleMatrix costDerivative(DoubleMatrix outputActivations, DoubleMatrix output) {
		return outputActivations.sub(output);
	}
}
