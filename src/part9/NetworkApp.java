package part9;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import mnist.MnistDataReader;
import mnist.MnistMatrix;

public class NetworkApp {

	private static final String FILE_DATA_60K = "data/train-images.idx3-ubyte";
	private static final String FILE_LABELS_60K = "data/train-labels.idx1-ubyte";
	private static final String FILE_DATA_10K = "data/t10k-images.idx3-ubyte";
	private static final String FILE_LABELS_10K = "data/t10k-labels.idx1-ubyte";

	private static List<double[][]> getTrainingDataFromMnist() throws IOException {
		List<double[][]> trainingData = new ArrayList<>();

		MnistMatrix[] mnistMatrix = new MnistDataReader().readData(FILE_DATA_10K, FILE_LABELS_10K);
		for (int i = 0; i < mnistMatrix.length; i++) {
			MnistMatrix matrix = mnistMatrix[i];
			double[][] io = new double[2][];
			double[] x = new double[784];

			for (int r = 0; r < matrix.getNumberOfRows(); r++) {
				for (int c = 0; c < matrix.getNumberOfColumns(); c++) {
					x[r * matrix.getNumberOfColumns() + c] = (double) matrix.getValue(r, c) / 255;
				}
			}
			double[] y = Stream.iterate(0, d -> d).limit(10).mapToDouble(d -> d).toArray();
			y[matrix.getLabel()] = 1;
			io[0] = x;
			io[1] = y;
			trainingData.add(io);
		}

		return trainingData;
	}

	public static void main(String[] args) throws IOException {
		List<double[][]> trainingData = getTrainingDataFromMnist();
		SigmoidNetwork net = new SigmoidNetwork(784, 30, 10);
		net.SGD(trainingData, 1000, 10, 3.0, trainingData);
	}

}
