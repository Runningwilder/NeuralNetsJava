package part9;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import mnist.MnistMatrix;

/**
 * Util class for serialization and other stuff. Not the best way to organize
 * methods but code conventions are discarded here
 * 
 */
public class Util {

	private Util() {

	}

	private static final String FILE_SERIALIZATION = "net.ser";

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
	 * Prints matrix
	 * @param matrix
	 */
    public static void printMnistMatrix(final MnistMatrix matrix) {
        System.out.println("label: " + matrix.getLabel());
        for (int r = 0; r < matrix.getNumberOfRows(); r++ ) {
            for (int c = 0; c < matrix.getNumberOfColumns(); c++) {
                System.out.print(matrix.getValue(r, c) + " ");
            }
            System.out.println();
        }
    }
}
