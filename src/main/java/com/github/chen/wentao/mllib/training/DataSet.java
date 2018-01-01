package com.github.chen.wentao.mllib.training;

import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DataSet implements MatrixWrapper {

	private final SimpleMatrix dataSet;

	public DataSet(SimpleMatrix theta) {
		this.dataSet = theta;
	}

	public static DataSet single(double... data) {
		return new DataSet(new SimpleMatrix(1, data.length, true, data));
	}

	public DataSet addPowerTerms(double... powers) {
		if (powers.length == 0) {
			return this;
		}
		SimpleMatrix dataSet = this.dataSet;
		for (double power : powers) {
			dataSet = dataSet.concatColumns(this.dataSet.elementPower(power));
		}
		return new DataSet(dataSet);
	}

	public DataSet addPowerTerms(List<Double> powers) {
		if (powers.isEmpty()) {
			return this;
		}
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

	public static class Builder {

		private final List<double[]> data = new ArrayList<>();

		public Builder add(double... data) {
			this.data.add(data);
			return this;
		}

		public DataSet build() {
			double[] flattenData = this.data.stream().flatMapToDouble(Arrays::stream).toArray();
			return new DataSet(new SimpleMatrix(data.size(), data.isEmpty() ? 0 : data.get(0).length, true, flattenData));
		}
	}
}
