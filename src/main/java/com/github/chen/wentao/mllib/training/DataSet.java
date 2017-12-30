package com.github.chen.wentao.mllib.training;

import org.ejml.simple.SimpleMatrix;

public class DataSet implements MatrixWrapper {

	private final SimpleMatrix dataSet;

	public DataSet(SimpleMatrix theta) {
		this.dataSet = theta;
	}

	public static DataSet single(double... data) {
		return new DataSet(new SimpleMatrix(1, data.length, true, data));
	}

	public DataSet addPowerTerms(double... powers) {
		SimpleMatrix dataSet = this.dataSet;
		for (double power : powers) {
			dataSet = dataSet.concatColumns(this.dataSet.elementPower(power));
		}
		return new DataSet(dataSet);
	}

	@Override
	public SimpleMatrix getMatrix() {
		return dataSet;
	}

	public int numFeatures() {
		return dataSet.numCols();
	}

	public int numExamples() {
		return dataSet.numRows();
	}
}
