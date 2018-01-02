package com.github.chen.wentao.mllib.training;

import com.github.chen.wentao.mllib.util.ejml.SimpleMatrixUtil;
import org.ejml.simple.SimpleMatrix;

import java.util.Arrays;

public class DataSetTarget implements MatrixWrapper {

	private final SimpleMatrix dataSet;
	private final SimpleMatrix binaryMatrix;
	private final int labels;

	public DataSetTarget(int... data) {
		this(new SimpleMatrix(data.length, 1, true, intArrayToDouble(data)), (int) Arrays.stream(data).distinct().count());
		assertValidValues();
	}

	public DataSetTarget(double... data) {
		this(new SimpleMatrix(data.length, 1, true, data), 0);
	}

	public DataSetTarget(SimpleMatrix theta, Integer labels) {
		if (theta.numCols() == 1) {
			this.dataSet = theta;
			this.labels = labels != null ? labels : countLabels();
			this.binaryMatrix = computeBinaryMatrix(theta, this.labels);
		} else {
			this.dataSet = theta;
			this.labels = theta.numCols();
			this.binaryMatrix = theta;
		}
	}

	@Override
	public SimpleMatrix getMatrix() {
		return dataSet;
	}

	public double get(int exampleIndex) {
		return dataSet.get(exampleIndex);
	}

	public int numExamples() {
		return dataSet.numRows();
	}

	public SimpleMatrix toBinaryMatrix() {
		return binaryMatrix;
	}

	private static SimpleMatrix computeBinaryMatrix(SimpleMatrix initMatrix, int labels) {
		SimpleMatrix targetBinaryMatrix = SimpleMatrixUtil.filterEquals(initMatrix, 0);
		for (int i = 1; i < labels; i++) {
			targetBinaryMatrix = targetBinaryMatrix.concatColumns(SimpleMatrixUtil.filterEquals(initMatrix, i));
		}
		return targetBinaryMatrix;
	}

	public int getLabel(int exampleIndex) {
		return (int) get(exampleIndex);
	}

	public int numLabels() {
		return labels;
	}

	private int countLabels() {
		int max = 0;
		SimpleMatrix matrix = getMatrix();
		for (int i = matrix.getNumElements() - 1; i >= 0; i--) {
			max = Math.max(max, (int) matrix.get(i));
		}
		return max + 1;
	}

	private static double[] intArrayToDouble(int[] data) {
		double[] array = new double[data.length];
		for (int i = 0; i < data.length; i++) {
			array[i] = data[i];
		}
		return array;
	}

	private void assertValidValues() {
		SimpleMatrix matrix = getMatrix();
		for (int i = matrix.getNumElements() - 1; i >= 0; i--) {
			assert(matrix.get(i) % 1.0 == 0.0);
		}
	}
}
