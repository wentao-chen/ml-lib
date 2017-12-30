package com.github.chen.wentao.mllib.data.scaling;

import com.github.chen.wentao.mllib.training.DataSet;
import com.github.chen.wentao.mllib.util.ejml.SimpleMatrixUtil;
import org.ejml.simple.SimpleMatrix;

public class FeatureMeanNormalizer implements FeatureScaler {

	private final SimpleMatrix columnMeans;
	private final SimpleMatrix columnRanges;
	private final DataSet normalizedDataSet;

	public FeatureMeanNormalizer(DataSet data) {
		SimpleMatrix dataMatrix = data.getMatrix();
		this.columnMeans = SimpleMatrixUtil.sumCols(dataMatrix).divide(data.numExamples());
		this.columnRanges = SimpleMatrixUtil.colMax(dataMatrix).minus(SimpleMatrixUtil.colMin(dataMatrix));
		this.normalizedDataSet = normalize(data);
	}

	@Override
	public DataSet getNormalizedDataSet() {
		return normalizedDataSet;
	}

	@Override
	public final DataSet normalize(DataSet data) {
		return new DataSet(normalize(data.getMatrix()));
	}

	private SimpleMatrix normalize(SimpleMatrix data) {
		assert(data.numCols() == columnMeans.numCols()); // Correct number of features

		boolean singleExample = data.numRows() == 1;

		SimpleMatrix ones = SimpleMatrixUtil.ones(data.numRows(), 1);
		SimpleMatrix means = singleExample ? columnMeans : ones.mult(columnMeans);
		SimpleMatrix ranges = singleExample ? columnRanges : ones.mult(columnRanges);
		return data.minus(means).elementDiv(ranges);
	}
}
