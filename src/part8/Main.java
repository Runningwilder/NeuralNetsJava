package part8;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Scanner;
import java.util.stream.Stream;

import org.jblas.DoubleMatrix;

/**
 * Util class for serialization and other stuff. Not the best way to organize
 * methods but code conventions are discarded here
 * 
 */
public class Main {

	private static final String FILE_SERIALIZATION = "net.ser";
	private static SigmoidNetwork net;

	/**
	 * Serializes object
	 * 
	 * @param obj object to serialize
	 * @throws IOException if anything strange with io occurred
	 */
	public static void serialize(Object obj) throws IOException {
		FileOutputStream fileOutputStream = new FileOutputStream(FILE_SERIALIZATION);
		try (ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream)) {
			objectOutputStream.writeObject(obj);
			objectOutputStream.flush();
		}
		System.out.println("Serialized");
	}

	/**
	 * Deserializes to object
	 * 
	 * @return deserialized object
	 * @throws IOException            if anything strange with io occurred
	 * @throws ClassNotFoundException if class wasn't found :(
	 */
	public static Object deserialize() throws IOException, ClassNotFoundException {
		Object obj;
		FileInputStream fileInputStream = new FileInputStream(FILE_SERIALIZATION);
		try (ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream)) {
			obj = objectInputStream.readObject();
		}
		return obj;
	}

	/**
	 * Converts in to DoubleMatrix
	 * 
	 * @param i int
	 * @return DoubleMatrix representation of int
	 */
	public static DoubleMatrix intToDoubleMatrix(int i) {
		double[] x = Stream.iterate(0, n -> 0).limit(256).mapToDouble(Double::new).toArray();
		x[i] = 1;
		return new DoubleMatrix(x);
	}

	/**
	 * Converts DoubleMatrix to String
	 * 
	 * @param dm DoubleMatrix
	 * @return String representation of DoubleMatrix
	 */
	public static String doubleMatrixToString(DoubleMatrix dm) {
		StringBuilder sb = new StringBuilder();
		for (double d : dm.toArray()) {
			sb.append(d >= 0.5 ? 1 : 0).append(' ');
		}
		return sb.toString();
	}
	
	public static void main(String[] args) throws ClassNotFoundException, IOException {
		net = (SigmoidNetwork) deserialize();
		try (Scanner sc = new Scanner(System.in)) {
			while (sc.hasNext()) {
				int input = Integer.parseInt(sc.next());
				System.out.println(doubleMatrixToString(net.feedForward(intToDoubleMatrix(input))));
			}
		}
	}

}
