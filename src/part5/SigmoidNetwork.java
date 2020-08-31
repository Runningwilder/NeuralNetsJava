package part5;

import org.jblas.DoubleMatrix;

public class SigmoidNetwork {

	private int numLayers;
	private int[] sizes;

	private DoubleMatrix[] weights;
	private DoubleMatrix[] biases;

	public SigmoidNetwork(int... sizes) {
		this.sizes = sizes;
		this.numLayers = sizes.length;

		this.biases = new DoubleMatrix[sizes.length - 1];
		this.weights = new DoubleMatrix[sizes.length - 1];

		// Storing biases
		for (int i = 1; i < sizes.length; i++) {
			double[][] temp = new double[sizes[i]][];
			for (int j = 0; j < sizes[i]; j++) {
				double[] b = new double[] { 1 }; // Set to a constant value for a while
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
					w[k] = 1; // Set to a constant value for a while
				}
				temp[j] = w;
			}
			weights[i - 1] = new DoubleMatrix(temp);
		}
//		final double  WEIGHT = 2;
//		double[][] customWeights = new double[][] {
//			{0, WEIGHT, 0, WEIGHT, 0, WEIGHT, 0, WEIGHT, 0, WEIGHT},
//			{0, 0, WEIGHT, WEIGHT, 0, 0, WEIGHT, WEIGHT, 0, 0},
//			{0, 0, 0, 0, WEIGHT, WEIGHT, WEIGHT, WEIGHT, 0, 0},
//			{0, 0, 0, 0, 0, 0, 0, 0, WEIGHT, WEIGHT}
//		};
//		weights[0] = new DoubleMatrix(customWeights);
		
	}

	public static void main(String[] args) {
		SigmoidNetwork net = new SigmoidNetwork(1, 1);
		double[] inputs = {0};
		double[] outputs = {0};
		double[][] inputsOuputs = new double[][] {inputs, outputs};
		DoubleMatrix[][] deltas = net.backProp(inputsOuputs);
		for(int i = 0; i < net.biases.length; i++) {
			net.biases[i] = net.biases[i].sub(deltas[0][i].mul(4));
		}
		System.out.println("Complete");
	}

	private DoubleMatrix[][] backProp(double[][] inputsOuputs) {
		DoubleMatrix[] nablaB = new DoubleMatrix[biases.length];
		DoubleMatrix[] nablaW = new DoubleMatrix[weights.length];
		
		for (int i = 0; i < nablaB.length; i++) {
			nablaB[i] = new DoubleMatrix(biases[i].getRows(), biases[i].getColumns());
		}
		for (int i = 0; i < nablaW.length; i++) {
			nablaW[i] = new DoubleMatrix(weights[i].getRows(), weights[i].getColumns());
		}
		
		//FeedForward
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
		
		//Backward pass
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
		return new DoubleMatrix[][] {nablaB, nablaW};
	}

	private DoubleMatrix sigmoidPrime(DoubleMatrix z) {
		return sigmoid(z).mul(sigmoid(z).rsub(1));
	}

	private DoubleMatrix costDerivative(DoubleMatrix outputActivations, DoubleMatrix output) {
		return outputActivations.sub(output);
	}

	/**
	 * 
	 * @param a - activation vector - the 1st layer also called the input layer
	 * @return DoubleMatrix - vector containing output from the network consisting
	 *         of float numbers between 0 and 1
	 */
	private DoubleMatrix feedForward(DoubleMatrix a) {
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
}