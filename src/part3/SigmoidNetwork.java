package part3;

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
					w[k] = 0; // Set to a constant value for a while
				}
				temp[j] = w;
			}
			weights[i - 1] = new DoubleMatrix(temp);
		}
	}

	public static void main(String[] args) {
		SigmoidNetwork net = new SigmoidNetwork(2, 3, 2);
		double[] inputs = { 1, 1 };
		DoubleMatrix outputs = net.feedForward(new DoubleMatrix(inputs));

		System.out.println(outputs.toString("%.4f"));
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