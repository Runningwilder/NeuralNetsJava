package part1;

public class PerceptronNetwork {

	private final int BIAS = 0;
	private final int WEIGHTS = 1;
	private int numLayers;
	private int[] sizes;

	public PerceptronNetwork(int... sizes) {
		this.sizes = sizes;
		this.numLayers = sizes.length;
	}

	public static void main(String[] args) {
		PerceptronNetwork net = new PerceptronNetwork(2, 3, 2);
		int[] inputs = { 1, 1 };
		int[] outputs = net.feedForward(inputs);

		for (int i = 0; i < outputs.length; i++) { // Prints output array consisting of 0s and 1s
			System.out.println(outputs[i]);
		}
	}

	/**
	 * 
	 * @param inputs - array of inputs - 1st layer also called the input layer
	 * @return array containing output from the network consisting of 0s and 1s
	 */
	private int[] feedForward(int[] inputs) {
		int[] outputs = null;
		for (int i = 1; i < numLayers; i++) { // Iterates over all layers (except first as it's the input layer)
			int size = sizes[i];
			int[] z = new int[size];
			outputs = new int[size];
			for (int j = 0; j < size; j++) { // Iterates over all neurons of layer i
				for (int k = 0; k < inputs.length; k++) { // Iterates over all inputs for layer i
					z[j] += WEIGHTS * inputs[k];
				}
				z[j] += BIAS;
				outputs[j] = z[j] > 0 ? 1 : 0;
			}
			inputs = outputs;
		}
		return outputs;
	}
}