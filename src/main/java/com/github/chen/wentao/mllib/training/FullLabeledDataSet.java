package com.github.chen.wentao.mllib.training;

import com.github.chen.wentao.mllib.util.ejml.SimpleMatrixUtil;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class FullLabeledDataSet extends FullDataSet {

	private final LabeledDataSetTarget dataSetTarget;

	public FullLabeledDataSet(DataSet dataSet, LabeledDataSetTarget dataSetTarget) {
		super(dataSet, dataSetTarget);
		this.dataSetTarget = dataSetTarget;
	}

	@Override
	public FullLabeledDataSet shuffle(Random random) {
		return matrixToFullDataSet(SimpleMatrixUtil.shuffleRows(getMatrix(), random));
	}

	public FullLabeledDataSet subset(int size) {
		SimpleMatrix matrix = getMatrix();
		return matrixToFullDataSet(matrix.extractMatrix(0, size, 0, matrix.numCols()));
	}

	private FullLabeledDataSet matrixToFullDataSet(SimpleMatrix m) {
		int features = getDataSet().numFeatures();
		SimpleMatrix dataMatrix = m.extractMatrix(0, m.numRows(), 0, features);
		SimpleMatrix targetMatrix = m.extractMatrix(0, m.numRows(), features, m.numCols());
		return new FullLabeledDataSet(new DataSet(dataMatrix), new LabeledDataSetTarget(targetMatrix));
	}

	@Override
	public LabeledDataSetTarget getDataSetTarget() {
		return dataSetTarget;
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

		public FullLabeledDataSet.Builder add(double... data) {
			this.data.add(Arrays.copyOfRange(data, 0, data.length - numOutputFeatures));
			this.dataTarget.add(Arrays.copyOfRange(data, data.length - numOutputFeatures, data.length));
			return this;
		}

		public FullLabeledDataSet build() {
			double[] flattenData = this.data.stream().flatMapToDouble(Arrays::stream).toArray();
			double[] flattenDataTarget = this.dataTarget.stream().flatMapToDouble(Arrays::stream).toArray();
			return new FullLabeledDataSet(
					new DataSet(new SimpleMatrix(data.size(), data.isEmpty() ? 0 : data.get(0).length, true, flattenData)),
					new LabeledDataSetTarget(new SimpleMatrix(dataTarget.size(), dataTarget.isEmpty() ? 0 : dataTarget.get(0).length, true, flattenDataTarget))
			);
		}
	}
}
