package part2;

public class SigmoidNetwork {

	private final double BIAS = -1.3;
	private final double WEIGHTS = 2.3;
	private int numLayers;
	private int[] sizes;

	public SigmoidNetwork(int... sizes) {
		this.sizes = sizes;
		this.numLayers = sizes.length;
	}

	public static void main(String[] args) {
		SigmoidNetwork net = new SigmoidNetwork(2, 3, 2);
		double[] inputs = { 1, 1 };
		double[] outputs = net.feedForward(inputs);

		for (int i = 0; i < outputs.length; i++) { // Prints output array consisting of float numbers between 0 and 1
			System.out.println(outputs[i]);
		}
	}

	/**
	 * 
	 * @param inputs - int[] of inputs - 1st layer also called the input layer
	 * @return int[] containing output from the network consisting of float numbers
	 *         between 0 and 1
	 */
	private double[] feedForward(double[] inputs) {
		double[] outputs = null;
		for (int i = 1; i < numLayers; i++) { // Iterates over all layers (except first as it's the input layer)
			int size = sizes[i];
			double[] z = new double[size];
			outputs = new double[size];
			for (int j = 0; j < size; j++) { // Iterates over all neurons of layer i
				for (int k = 0; k < inputs.length; k++) { // Iterates over all inputs for layer i
					z[j] += WEIGHTS * inputs[k];
				}
				z[j] += BIAS;
				outputs[j] = 1 / (1 + Math.exp(-z[j]));
			}
			inputs = outputs;
		}
		return outputs;
	}
}