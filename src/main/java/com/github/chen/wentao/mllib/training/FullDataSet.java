package com.github.chen.wentao.mllib.training;

import com.github.chen.wentao.mllib.util.ejml.SimpleMatrixUtil;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class FullDataSet implements MatrixWrapper {

	private final DataSet dataSet;
	private final DataSetTarget dataSetTarget;
	private final SimpleMatrix matrix;

	public FullDataSet(DataSet dataSet, DataSetTarget dataSetTarget) {
		this.dataSet = dataSet;
		this.dataSetTarget = dataSetTarget;
		this.matrix = dataSet.getMatrix().concatColumns(dataSetTarget.getMatrix());
	}

	public DataSet getDataSet() {
		return dataSet;
	}

	public DataSetTarget getDataSetTarget() {
		return dataSetTarget;
	}

	public FullDataSet shuffle(Random random) {
		return matrixToFullDataSet(SimpleMatrixUtil.shuffleRows(matrix, random));
	}

	public FullDataSet subset(int size) {
		return matrixToFullDataSet(matrix.extractMatrix(0, size, 0, matrix.numCols()));
	}

	private FullDataSet matrixToFullDataSet(SimpleMatrix m) {
		SimpleMatrix dataMatrix = m.extractMatrix(0, m.numRows(), 0, dataSet.numFeatures());
		SimpleMatrix targetMatrix = m.extractMatrix(0, m.numRows(), dataSet.numFeatures(), m.numCols());
		return new FullDataSet(new DataSet(dataMatrix), new DataSetTarget(targetMatrix));
	}

	@Override
	public SimpleMatrix getMatrix() {
		return matrix;
	}

	public static class Builder {

		private final int numOutputFeatures;
		private final List<double[]> data = new ArrayList<>();
		private final List<double[]> dataTarget = new ArrayList<>();

		public Builder() {
			this(1);
		}

		public Builder(int numOutputFeatures) {
			this.numOutputFeatures = numOutputFeatures;
		}

		public FullDataSet.Builder add(double... data) {
			this.data.add(Arrays.copyOfRange(data, 0, data.length - numOutputFeatures));
			this.dataTarget.add(Arrays.copyOfRange(data, data.length - numOutputFeatures, data.length));
			return this;
		}

		public FullDataSet build() {
			double[] flattenData = this.data.stream().flatMapToDouble(Arrays::stream).toArray();
			double[] flattenDataTarget = this.dataTarget.stream().flatMapToDouble(Arrays::stream).toArray();
			return new FullDataSet(
					new DataSet(new SimpleMatrix(data.size(), data.isEmpty() ? 0 : data.get(0).length, true, flattenData)),
					new LabeledDataSetTarget(new SimpleMatrix(dataTarget.size(), dataTarget.isEmpty() ? 0 : dataTarget.get(0).length, true, flattenDataTarget))
			);
		}
	}
}
