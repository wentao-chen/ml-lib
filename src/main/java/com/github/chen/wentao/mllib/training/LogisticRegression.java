package com.github.chen.wentao.mllib.training;

import com.github.chen.wentao.mllib.data.DataUtil;
import com.github.chen.wentao.mllib.util.ejml.SimpleMatrixUtil;
import org.ejml.simple.SimpleMatrix;

import static com.github.chen.wentao.mllib.util.ejml.SimpleMatrixUtil.ones;

public class LogisticRegression {

	private static final double SIGMOID_SCALE = Math.nextDown(1.0);
	private static final double SIGMOID_OFFSET = Double.MIN_VALUE;

	public static double sigmoid(double z) {
		return 1.0 / (1.0 + Math.exp(-z));
	}

	/**
	 * Performs the sigmoid function on every element of a matrix
	 * @param m the matrix
	 * @return a new matrix where each element is the sigmoid of the corresponding element in matrix {@code m}
	 */
	public static SimpleMatrix sigmoid(SimpleMatrix m) {
		return m.negative().elementExp().plus(1.0).elementPower(-1.0);
	}

	/**
	 * Calculates the hypothesis value for a data set given parameters theta
	 * @param theta (n + 1) vector of n parameter features (and bias parameter)
	 * @param dataSet (m) x (n + 1) matrix of m training examples and n features (and bias - first column vector should only 1s)
	 * @return m vector of the hypothesis value for each training example
	 */
	public static DataSetTarget hypothesis(FeatureParameters theta, DataSet dataSet) {
		return new DataSetTarget(hypothesis(theta.getMatrix(), dataSet.getMatrix()));
	}

	/**
	 * Calculates the hypothesis value for a data set given parameters theta
	 * @param theta (n + 1) vector of n parameter features (and bias parameter)
	 * @param dataSet (m) x (n + 1) matrix of m training examples and n features (and bias - first column vector should only 1s)
	 * @return m vector of the hypothesis value for each training example
	 */
	private static SimpleMatrix hypothesis(SimpleMatrix theta, SimpleMatrix dataSet) {
		assert(theta.numCols() == 1); // is vector
		assert(theta.numRows() == dataSet.numCols()); // correct number of features

		return sigmoid(dataSet.mult(theta));
	}

	/**
	 * Calculates the cost for a data set given parameters theta
	 * @param theta (n + 1) vector of n parameter features (and bias parameter)
	 * @param dataSet (m) x (n) matrix of m training examples and n features
	 * @param target (m) vector of the target values for each m training examples (each element should be either 0 or 1)
	 * @param lambda the regularization parameter (greater or equal to 0)
	 * @return double of the cost of the parameters for the data set
	 */
	public static double costFunction(FeatureParameters theta, DataSet dataSet, DataSetTarget target, double lambda) {
		return costFunction(DataUtil.addBiasColumn(theta.getMatrix()), dataSet.getMatrix(), target.getMatrix(), lambda);
	}

	/**
	 * Calculates the cost for a data set given parameters theta
	 * @param theta (n + 1) vector of n parameter features (and bias parameter)
	 * @param dataSet (m) x (n + 1) matrix of m training examples and n features
	 * @param target (m) vector of the target values for each m training examples (each element should be either 0 or 1)
	 * @param lambda the regularization parameter (greater or equal to 0)
	 * @return double of the cost of the parameters for the data set
	 */
	private static double costFunction(SimpleMatrix theta, SimpleMatrix dataSet, SimpleMatrix target, double lambda) {
		assert(theta.numCols() == 1); // is vector
		assert(theta.numRows() == dataSet.numCols()); // correct number of features
		assert(dataSet.numRows() == target.numRows()); // correct number of training examples
		assert(target.numCols() == 1); // is vector
		assertValidTargetValues(target, 2);
		assert(lambda >= 0 && Double.isFinite(lambda));

		double m = dataSet.numRows();
		SimpleMatrix hypothesis = hypothesis(theta, dataSet)
				.scale(SIGMOID_SCALE).plus(SIGMOID_OFFSET); // Avoid sigmoid function from outputting 0.0 or 1.0 which may produce NaN when taking logarithms;
		SimpleMatrix temp1 = target.negative().elementMult(hypothesis.elementLog());
		SimpleMatrix temp2 = ones(target).minus(target).elementMult(ones(hypothesis).minus(hypothesis).elementLog());
		double cost = temp1.minus(temp2).elementSum();
		double regularizationCost = lambda == 0.0 ? 0.0 : lambda / 2.0 * theta.extractMatrix(1, theta.numRows(), 0, 1).elementPower(2.0).elementSum();
		return (cost + regularizationCost) / m;
	}

	/**
	 * Calculates the cost for a data set given parameters theta
	 * @param thetas array of (n + 1) vectors of n parameter features (and bias parameter) for each label
	 * @param dataSet (m) x (n) matrix of m training examples and n features
	 * @param target (m) vector of the target values for each m training examples (each element should be either 0 or 1)
	 * @param lambda the regularization parameter (greater or equal to 0)
	 * @return double of the cost of the parameters for the data set
	 */
	public static double costFunctionMulti(FeatureParameters[] thetas, DataSet dataSet, LabeledDataSetTarget target, double lambda) {
		return costFunctionMulti(convert(thetas), dataSet.getMatrix(), target.getMatrix(), lambda);
	}

	/**
	 * Calculates the cost for a data set given parameters theta
	 * @param thetas array of (n + 1) vectors of n parameter features (and bias parameter) for each label
	 * @param dataSet (m) x (n) matrix of m training examples and n features
	 * @param target (m) vector of the target values for each m training examples (each element should be either 0 or 1)
	 * @param lambda the regularization parameter (greater or equal to 0)
	 * @return double of the cost of the parameters for the data set
	 */
	private static double costFunctionMulti(SimpleMatrix[] thetas, SimpleMatrix dataSet, SimpleMatrix target, double lambda) {
		for (SimpleMatrix theta : thetas) {
			assert(theta.numCols() == 1); // is vector
			assert(theta.numRows() == dataSet.numCols() + 1); // correct number of features
		}
		assert(dataSet.numRows() == target.numRows()); // correct number of training examples
		assert(target.numCols() == 1); // is vector
		assertValidTargetValues(target, thetas.length);

		// Add bias values
		dataSet = DataUtil.addBiasColumn(dataSet);

		double totalCost = 0;
		for (int label = 0; label < thetas.length; label++) {
			totalCost += costFunction(thetas[label], dataSet, SimpleMatrixUtil.filterEquals(target, label), lambda);
		}
		return totalCost / thetas.length;
	}

	/**
	 * Predicts whether each data example is 0 or 1 using parameters theta with a threshold at 0.5
	 * @param theta (n + 1) vector of n parameter features (and bias parameter)
	 * @param dataSet (m) x (n + 1) matrix of m training examples and n features (and bias - first column vector should only 1s)
	 * @return m vector of the hypothesis value for each training example
	 */
	public static LabeledDataSetTarget predict(FeatureParameters theta, DataSet dataSet) {
		return new LabeledDataSetTarget(predict(theta.getMatrix(), dataSet.getMatrix()));
	}

	/**
	 * Predicts whether each data example is 0 or 1 using parameters theta with a threshold at 0.5
	 * @param theta (n + 1) vector of n parameter features (and bias parameter)
	 * @param dataSet (m) x (n + 1) matrix of m training examples and n features (and bias - first column vector should only 1s)
	 * @return m vector of the hypothesis value for each training example
	 */
	private static SimpleMatrix predict(SimpleMatrix theta, SimpleMatrix dataSet) {
		assert(theta.numCols() == 1); // is vector
		assert(theta.numRows() == dataSet.numCols()); // correct number of features

		SimpleMatrix product = dataSet.mult(theta);
		double[] predictions = new double[product.numRows()];
		for (int i = 0; i < predictions.length; i++) {
			predictions[i] = product.get(i) >= 0 ? 1.0 : 0.0;
		}
		return new SimpleMatrix(predictions.length, 1, true, predictions);
	}

	/**
	 * Predicts whether each data example is 0 or 1 using parameters theta with a threshold at 0.5
	 * @param thetas array of (n + 1) vectors of n parameter features (and bias parameter) for each label
	 * @param dataSet (m) x (n + 1) matrix of m training examples and n features (and bias - first column vector should only 1s)
	 * @return m vector of the hypothesis value for each training example
	 */
	public static LabeledDataSetTarget predictMulti(FeatureParameters[] thetas, DataSet dataSet) {
		return new LabeledDataSetTarget(predictMulti(convert(thetas), DataUtil.addBiasColumn(dataSet.getMatrix())));
	}

	/**
	 * Predicts whether each data example is 0 or 1 using parameters theta with a threshold at 0.5
	 * @param thetas array of (n + 1) vectors of n parameter features (and bias parameter) for each label
	 * @param dataSet (m) x (n + 1) matrix of m training examples and n features (and bias - first column vector should only 1s)
	 * @return m vector of the hypothesis value for each training example
	 */
	private static SimpleMatrix predictMulti(SimpleMatrix[] thetas, SimpleMatrix dataSet) {
		for (SimpleMatrix theta : thetas) {
			assert(theta.numCols() == 1); // is vector
			assert(theta.numRows() == dataSet.numCols()); // correct number of features
		}

		SimpleMatrix[] products = new SimpleMatrix[thetas.length];
		for (int i = 0; i < products.length; i++) {
			products[i] = dataSet.mult(thetas[i]);
		}
		double[] predictions = new double[dataSet.numRows()];
		for (int i = 0; i < predictions.length; i++) {
			int maxLabel = 0;
			double maxLabelValue = products[0].get(i);
			for (int label = 1; label < products.length; label++) {
				double value = products[label].get(i);
				if (value > maxLabelValue) {
					maxLabel = label;
					maxLabelValue = value;
				}
			}
			predictions[i] = maxLabel;
		}
		return new SimpleMatrix(predictions.length, 1, true, predictions);
	}

	/**
	 * Predicts whether each data example is 0 or 1 using parameters theta
	 * @param theta (n + 1) vector of n parameter features (and bias parameter)
	 * @param dataSet (m) x (n + 1) matrix of m training examples and n features (and bias - first column vector should only 1s)
	 * @param threshold double in range [0.0, 1.0] of the minimum sigmoid value which is classified with 1
	 * @return m vector of the hypothesis value for each training example
	 */
	public static LabeledDataSetTarget predict(FeatureParameters theta, DataSet dataSet, double threshold) {
		return new LabeledDataSetTarget(predict(theta.getMatrix(), dataSet.getMatrix(), threshold));
	}

	/**
	 * Predicts whether each data example is 0 or 1 using parameters theta
	 * @param theta (n + 1) vector of n parameter features (and bias parameter)
	 * @param dataSet (m) x (n + 1) matrix of m training examples and n features (and bias - first column vector should only 1s)
	 * @param threshold double in range [0.0, 1.0] of the minimum sigmoid value which is classified with 1
	 * @return m vector of the hypothesis value for each training example
	 */
	private static SimpleMatrix predict(SimpleMatrix theta, SimpleMatrix dataSet, double threshold) {
		assert(theta.numCols() == 1); // is vector
		assert(theta.numRows() == dataSet.numCols()); // correct number of features

		SimpleMatrix product = dataSet.mult(theta);
		double[] predictions = new double[product.numRows()];
		for (int i = 0; i < predictions.length; i++) {
			predictions[i] = sigmoid(product.get(i)) >= threshold ? 1.0 : 0.0;
		}
		return new SimpleMatrix(predictions.length, 1, true, predictions);
	}

	/**
	 * Performs gradient descent to find the optimal parameters theta which minimizes the cost function for a data set.
	 * @param dataSet (m) x (n) matrix of m training examples and n features
	 * @param target (m) vector of the target values for each m training examples (each element should be either 0 or 1)
	 * @param initialTheta (n + 1) vector of initial n parameter features (and bias parameter)
	 * @param alpha the learning rate (greater than 0)
	 * @param lambda the regularization parameter (greater or equal to 0)
	 * @param numIterations the maximum number of iterations to be performed
	 * @return n + 1 vector of the optimal parameters theta
	 */
	public static FeatureParameters gradientDescent(DataSet dataSet, LabeledDataSetTarget target, FeatureParameters initialTheta, double alpha, double lambda, int numIterations) {
		return new FeatureParameters(gradientDescent(DataUtil.addBiasColumn(dataSet.getMatrix()), target.getMatrix(), initialTheta.getMatrix(), alpha, lambda, numIterations));
	}

	/**
	 * Performs gradient descent to find the optimal parameters theta which minimizes the cost function for a data set.
	 * @param dataSet (m) x (n + 1) matrix of m training examples and n features
	 * @param target (m) vector of the target values for each m training examples (each element should be either 0 or 1)
	 * @param initialTheta (n + 1) vector of initial n parameter features (and bias parameter)
	 * @param alpha the learning rate (greater than 0)
	 * @param lambda the regularization parameter (greater or equal to 0)
	 * @param numIterations the maximum number of iterations to be performed
	 * @return n + 1 vector of the optimal parameters theta
	 */
	private static SimpleMatrix gradientDescent(SimpleMatrix dataSet, SimpleMatrix target, SimpleMatrix initialTheta, double alpha, double lambda, int numIterations) {
		assert(initialTheta.numCols() == 1); // is vector
		assert(initialTheta.numRows() == dataSet.numCols()); // correct number of features
		assert(dataSet.numRows() == target.numRows()); // correct number of training examples
		assert(target.numCols() == 1); // is vector
		assertValidTargetValues(target, 2);
		assert(alpha > 0 && Double.isFinite(alpha));

		double m = dataSet.numRows();
		double scale = alpha / m;
		double regularizationScale = 1.0 - scale * lambda;
		SimpleMatrix theta = initialTheta;
		for (int i = 0; i < numIterations; i++) {
			SimpleMatrix previousTheta;
			if (lambda == 0.0) {
				previousTheta = theta; // No regularization
			} else {
				double previousBias = theta.get(0);
				previousTheta = theta.scale(regularizationScale); // With regularization
				previousTheta.set(0, previousBias); // By convention, bias term is not penalized
			}
			theta = previousTheta.minus(dataSet.transpose().mult(hypothesis(theta, dataSet).minus(target)).scale(scale));
		}
		return theta;
	}

	/**
	 * Performs gradient descent to find the optimal parameters theta which minimizes the cost function for a data set.
	 * Trains {@code labels} number of classifiers and uses one-vs-all strategy to perform multi-class classification
	 * @param dataSet (m) x (n) matrix of m training examples and n features
	 * @param target (m) vector of the target values for each m training examples (each element should be either an integer greater or equal to 0 and less than the number of labels)
	 * @param initialTheta (n + 1) vector of initial n parameter features (and bias parameter)
	 * @param alpha the learning rate (greater than 0)
	 * @param lambda the regularization parameter (greater or equal to 0)
	 * @param numIterations the maximum number of iterations to be performed
	 * @return (number of labels)-length array of n + 1 vectors of the optimal parameters theta
	 */
	public static FeatureParameters[] gradientDescentMulti(DataSet dataSet, LabeledDataSetTarget target, FeatureParameters initialTheta, double alpha, double lambda, int numIterations) {
		SimpleMatrix[] labelGradientDescents = gradientDescentMulti(dataSet.getMatrix(), target.getMatrix(), initialTheta.getMatrix(), target.numLabels(), alpha, lambda, numIterations);
		FeatureParameters[] labelFeatureParameters = new FeatureParameters[labelGradientDescents.length];
		for (int i = 0; i < labelFeatureParameters.length; i++) {
			labelFeatureParameters[i] = new FeatureParameters(labelGradientDescents[i]);
		}
		return labelFeatureParameters;
	}

	/**
	 * Performs gradient descent to find the optimal parameters theta which minimizes the cost function for a data set.
	 * Trains {@code labels} number of classifiers and uses one-vs-all strategy to perform multi-class classification
	 * @param dataSet (m) x (n) matrix of m training examples and n features
	 * @param target (m) vector of the target values for each m training examples (each element should be either an integer greater or equal to 0 and less than the number of labels)
	 * @param initialTheta (n + 1) vector of initial n parameter features (and bias parameter)
	 * @param labels the number of different labels used
	 * @param alpha the learning rate (greater than 0)
	 * @param lambda the regularization parameter (greater or equal to 0)
	 * @param numIterations the maximum number of iterations to be performed
	 * @return (number of labels)-length array of n + 1 vectors of the optimal parameters theta
	 */
	private static SimpleMatrix[] gradientDescentMulti(SimpleMatrix dataSet, SimpleMatrix target, SimpleMatrix initialTheta, int labels, double alpha, double lambda, int numIterations) {
		assert(initialTheta.numCols() == 1); // is vector
		assert(initialTheta.numRows() == dataSet.numCols() + 1); // correct number of features
		assert(dataSet.numRows() == target.numRows()); // correct number of training examples
		assert(target.numCols() == 1); // is vector
		assertValidTargetValues(target, labels);
		assert(alpha > 0 && Double.isFinite(alpha));

		// Add bias values
		dataSet = DataUtil.addBiasColumn(dataSet);

		SimpleMatrix[] labelGradientDescents = new SimpleMatrix[labels];
		for (int i = 0; i < labels; i++) {
			SimpleMatrix binaryTarget = SimpleMatrixUtil.filterEquals(target, i);
			labelGradientDescents[i] = gradientDescent(dataSet, binaryTarget, initialTheta, alpha, lambda, numIterations);
		}
		return labelGradientDescents;
	}

	private static void assertValidTargetValues(SimpleMatrix target, int labels) {
		for (int i = target.getNumElements() - 1; i >= 0; i--) {
			double value = target.get(i);
			assert(value % 1.0 == 0.0 && value >= 0.0 && value < labels);
		}
	}

	private static SimpleMatrix[] convert(MatrixWrapper[] matrixWrappers) {
		SimpleMatrix[] matrices = new SimpleMatrix[matrixWrappers.length];
		for (int i = 0; i < matrixWrappers.length; i++) {
			matrices[i] = matrixWrappers[i].getMatrix();
		}
		return matrices;
	}
}
