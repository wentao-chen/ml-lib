package com.github.chen.wentao.mllib.data;

import org.ejml.simple.SimpleMatrix;

import static com.github.chen.wentao.mllib.util.ejml.SimpleMatrixUtil.ones;

public class DataUtil {

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

	public static double sigmoidGrad(double z) {
		double sigmoid = sigmoid(z);
		return sigmoid * (1.0 - sigmoid);
	}

	/**
	 * Computes the derivative of the sigmoid function on every element of a matrix
	 * @param m the matrix
	 * @return a new matrix where each element is the sigmoid derivative of the corresponding element in matrix {@code m}
	 */
	public static SimpleMatrix sigmoidGrad(SimpleMatrix m) {
		SimpleMatrix sigmoid = sigmoid(m);
		return sigmoid.elementMult(ones(sigmoid).minus(sigmoid));
	}

	public static SimpleMatrix addBiasColumn(SimpleMatrix data) {
		return ones(data.numRows(), 1).concatColumns(data);
	}

	public static SimpleMatrix addBiasRow(SimpleMatrix data) {
		return ones(1, data.numCols()).concatRows(data);
	}
}
