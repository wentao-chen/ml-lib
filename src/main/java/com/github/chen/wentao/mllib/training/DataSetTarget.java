package com.github.chen.wentao.mllib.training;

import org.ejml.simple.SimpleMatrix;

public class DataSetTarget implements MatrixWrapper {

	private final SimpleMatrix dataSet;

	public DataSetTarget(double... data) {
		this(new SimpleMatrix(data.length, 1, true, data));
	}

	public DataSetTarget(SimpleMatrix theta) {
		assert(theta.numCols() == 1); // is vector

		this.dataSet = theta;
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
}
