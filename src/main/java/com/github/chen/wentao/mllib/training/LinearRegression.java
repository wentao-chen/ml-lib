package com.github.chen.wentao.mllib.training;

import com.github.chen.wentao.mllib.data.DataUtil;
import com.github.chen.wentao.mllib.util.ejml.SimpleMatrixUtil;
import org.ejml.simple.SimpleMatrix;

public class LinearRegression {

	public static SupervisedLearningAlgorithm<FeatureParameters> getAlgorithm(FeatureParameters initial, double alpha, double lambda, int numIterations) {
		return (dataSet, target) -> LinearRegression.gradientDescent(dataSet, target, initial, alpha, lambda, numIterations);
	}

	public static CostFunction<FeatureParameters> getCostFunction(double lambda) {
		return (theta, dataSet, target) -> costFunction(theta, dataSet, target, lambda);
	}

	/**
	 * Calculates the hypothesis value for a data set given parameters theta
	 *
	 * @param theta   (n + 1) vector of n parameter features (and bias parameter)
	 * @param dataSet (m) x (n + 1) matrix of m training examples and n features (and bias - first column vector should only 1s)
	 * @return m vector of the hypothesis value for each training example
	 */
	public static DataSetTarget hypothesis(FeatureParameters theta, DataSet dataSet) {
		return new DataSetTarget(hypothesis(theta.getMatrix(), dataSet.getMatrix()), 0);
	}

	/**
	 * Calculates the hypothesis value for a data set given parameters theta
	 *
	 * @param theta   (n + 1) vector of n parameter features (and bias parameter)
	 * @param dataSet (m) x (n + 1) matrix of m training examples and n features (and bias - first column vector should only 1s)
	 * @return m vector of the hypothesis value for each training example
	 */
	private static SimpleMatrix hypothesis(SimpleMatrix theta, SimpleMatrix dataSet) {
		assert (theta.numCols() == 1); // is vector
		assert (theta.numRows() == dataSet.numCols()); // correct number of features

		return dataSet.mult(theta);
	}

	/**
	 * Calculates the cost for a data set given parameters theta
	 *
	 * @param theta   (n + 1) vector of n parameter features (and bias parameter)
	 * @param dataSet (m) x (n) matrix of m training examples and n features
	 * @param target  (m) vector of the target values for each m training examples
	 * @param lambda  the regularization parameter (greater or equal to 0)
	 * @return double of the cost of the parameters for the data set
	 */
	public static double costFunction(FeatureParameters theta, DataSet dataSet, DataSetTarget target, double lambda) {
		return costFunction(theta.getMatrix(), DataUtil.addBiasColumn(dataSet.getMatrix()), target.getMatrix(), lambda);
	}

	/**
	 * Calculates the cost for a data set given parameters theta
	 *
	 * @param theta   (n + 1) vector of n parameter features (and bias parameter)
	 * @param dataSet (m) x (n + 1) matrix of m training examples and n features (and bias value)
	 * @param target  (m) vector of the target values for each m training examples
	 * @param lambda  the regularization parameter (greater or equal to 0)
	 * @return double of the cost of the parameters for the data set
	 */
	private static double costFunction(SimpleMatrix theta, SimpleMatrix dataSet, SimpleMatrix target, double lambda) {
		assert (theta.numCols() == 1); // is vector
		assert (theta.numRows() == dataSet.numCols()); // correct number of features
		assert (dataSet.numRows() == target.numRows()); // correct number of training examples
		assert (target.numCols() == 1); // is vector
		assert (lambda >= 0 && Double.isFinite(lambda));

		double m = dataSet.numRows();
		double cost = hypothesis(theta, dataSet).minus(target).elementPower(2.0).elementSum();
		double regularizationCost = lambda == 0.0 ? 0.0 : lambda * theta.extractMatrix(1, theta.numRows(), 0, 1).elementPower(2.0).elementSum();
		return (cost + regularizationCost) / (2.0 * m);
	}

	/**
	 * Performs gradient descent to find the optimal parameters theta which minimizes the cost function for a data set.
	 * <i>Recommended over normal equation for large number of features (m ~ 1000 to 10000).</i>
	 *
	 * @param dataSet (m) x (n) matrix of m training examples and n features
	 * @param target (m) vector of the target values for each m training examples
	 * @param initialTheta (n + 1) vector of initial n parameter features (and bias parameter)
	 * @param alpha the learning rate (greater than 0)
	 * @param lambda the regularization parameter (greater or equal to 0)
	 * @param numIterations the maximum number of iterations to be performed
	 * @return n + 1 vector of the optimal parameters theta
	 * @see #normalEquation(DataSet, DataSetTarget, double)
	 */
	public static FeatureParameters gradientDescent(DataSet dataSet, DataSetTarget target, FeatureParameters initialTheta, double alpha, double lambda, int numIterations) {
		return new FeatureParameters(gradientDescent(DataUtil.addBiasColumn(dataSet.getMatrix()), target.getMatrix(), initialTheta.getMatrix(), alpha, lambda, numIterations));
	}

	/**
	 * Performs gradient descent to find the optimal parameters theta which minimizes the cost function for a data set.
	 * <i>Recommended over normal equation for large number of features (m ~ 1000 to 10000).</i>
	 *
	 * @param dataSet (m) x (n + 1) matrix of m training examples and n features (and bias value)
	 * @param target (m) vector of the target values for each m training examples
	 * @param initialTheta (n + 1) vector of initial n parameter features (and bias parameter)
	 * @param alpha the learning rate (greater than 0)
	 * @param lambda the regularization parameter (greater or equal to 0)
	 * @param numIterations the maximum number of iterations to be performed
	 * @return n + 1 vector of the optimal parameters theta
	 * @see #normalEquation(SimpleMatrix, SimpleMatrix, double)
	 */
	private static SimpleMatrix gradientDescent(SimpleMatrix dataSet, SimpleMatrix target, SimpleMatrix initialTheta, double alpha, double lambda, int numIterations) {
		assert(initialTheta.numCols() == 1); // is vector
		assert(initialTheta.numRows() == dataSet.numCols()); // correct number of features
		assert(dataSet.numRows() == target.numRows()); // correct number of training examples
		assert(target.numCols() == 1); // is vector
		assert(alpha > 0 && Double.isFinite(alpha));
		assert(lambda >= 0 && Double.isFinite(lambda));

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
	 * Solves for the optimal parameters theta which minimizes the cost function for a data set.
	 * <i>Recommended over gradient descent for small number of features (n ~ 1000 to 10000).</i>
	 *
	 * @param dataSet (m) x (n) matrix of m training examples and n features
	 * @param target (m) vector of the target values for each m training examples
	 * @param lambda the regularization parameter (greater or equal to 0)
	 * @return n + 1 vector of the optimal parameters theta
	 * @see #gradientDescent(DataSet, DataSetTarget, FeatureParameters, double, double, int)
	 */
	public static FeatureParameters normalEquation(DataSet dataSet, DataSetTarget target, double lambda) {
		return new FeatureParameters(normalEquation(dataSet.getMatrix(), target.getMatrix(), lambda));
	}

	/**
	 * Solves for the optimal parameters theta which minimizes the cost function for a data set.
	 * <i>Recommended over gradient descent for small number of features (n ~ 1000 to 10000).</i>
	 *
	 * @param dataSet (m) x (n) matrix of m training examples and n features
	 * @param target (m) vector of the target values for each m training examples
	 * @param lambda the regularization parameter (greater or equal to 0)
	 * @return n + 1 vector of the optimal parameters theta
	 * @see #gradientDescent(SimpleMatrix, SimpleMatrix, SimpleMatrix, double, double, int)
	 */
	private static SimpleMatrix normalEquation(SimpleMatrix dataSet, SimpleMatrix target, double lambda) {
		assert(dataSet.numRows() == target.numRows()); // correct number of training examples
		assert(target.numCols() == 1); // is vector
		assert(lambda >= 0 && Double.isFinite(lambda));

		// Add bias values
		dataSet = DataUtil.addBiasColumn(dataSet);

		SimpleMatrix dataSetTranspose = dataSet.transpose();
		SimpleMatrix temp = dataSetTranspose.mult(dataSet);
		if (lambda > 0) {
			temp = temp.plus(SimpleMatrixUtil.shortDiagonal(temp.numCols(), lambda)); // With regularization
		}
		return temp.pseudoInverse().mult(dataSetTranspose).mult(target);
	}
}
