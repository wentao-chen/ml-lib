package com.github.chen.wentao.mllib.data.scaling;

import com.github.chen.wentao.mllib.training.DataSet;
import com.github.chen.wentao.mllib.util.ejml.SimpleMatrixUtil;
import org.ejml.simple.SimpleMatrix;

public class FeatureRescaler implements FeatureScaler {

	private final SimpleMatrix columnMin;
	private final SimpleMatrix columnRanges;
	private final DataSet normalizedDataSet;

	public FeatureRescaler(DataSet data) {
		SimpleMatrix dataMatrix = data.getMatrix();
		this.columnMin = SimpleMatrixUtil.colMin(dataMatrix);
		this.columnRanges = SimpleMatrixUtil.colMax(dataMatrix).minus(columnMin);
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
		assert(data.numCols() == columnMin.numCols()); // Correct number of features

		boolean singleExample = data.numRows() == 1;

		SimpleMatrix ones = SimpleMatrixUtil.ones(data.numRows(), 1);
		SimpleMatrix min = singleExample ? columnMin : ones.mult(columnMin);
		SimpleMatrix ranges = singleExample ? columnRanges : ones.mult(columnRanges);
		return data.minus(min).elementDiv(ranges);
	}
}
