package part9;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Stream;

public class NetworkApp {

//	public static void main(String[] args) throws ClassNotFoundException, IOException {
//		net = (SigmoidNetwork) deserialize();
//		try (Scanner sc = new Scanner(System.in)) {
//			while (sc.hasNext()) {
//				int input = Integer.parseInt(sc.next());
//				System.out.println(doubleMatrixToString(net.feedForward(intToDoubleMatrix(input))));
//			}
//		}
//	}
	

	public static void main(String[] args) {
		List<double[][]> inputsOutputs = new ArrayList<>();
		for (int i = 0; i < 256; i++) {
			double[][] io = new double[2][];
			double[] x = new double[256];
			double[] y = new double[8];

			String binary = String.format("%8s", Integer.toBinaryString(i)).replace(' ', '0');
			x = Stream.iterate(0, n -> 0).limit(256).mapToDouble(Double::new).toArray();
			y = Arrays.stream(binary.split("")).mapToDouble(Double::parseDouble).toArray();

			x[i] = 1;
			io[0] = x;
			io[1] = y;
			inputsOutputs.add(io);
		}
		SigmoidNetwork net = new SigmoidNetwork(256, 32, 8);
		// We're training the net here with the set of all possible data combinations.
		// It's only for educational purposes
		net.SGD(inputsOutputs, 1000, 8, 15, inputsOutputs);
	}
	
}
