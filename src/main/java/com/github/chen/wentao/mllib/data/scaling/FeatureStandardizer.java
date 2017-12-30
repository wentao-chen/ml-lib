package com.github.chen.wentao.mllib.data.scaling;

import com.github.chen.wentao.mllib.training.DataSet;
import com.github.chen.wentao.mllib.util.ejml.SimpleMatrixUtil;
import org.ejml.simple.SimpleMatrix;

public class FeatureStandardizer implements FeatureScaler {

	private final SimpleMatrix columnMeans;
	private final SimpleMatrix columnStandardDeviations;
	private final DataSet normalizedDataSet;

	public FeatureStandardizer(DataSet data) {
		double m = data.numExamples();
		SimpleMatrix dataMatrix = data.getMatrix();
		SimpleMatrix columnSums = SimpleMatrixUtil.sumCols(dataMatrix);
		SimpleMatrix columnSquaredSums = SimpleMatrixUtil.sumCols(dataMatrix.elementPower(2.0));
		this.columnMeans = columnSums.divide(m);
		this.columnStandardDeviations = columnSquaredSums.divide(m).minus(columnMeans.elementPower(2.0)).elementPower(0.5);
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
		SimpleMatrix standardDeviations = singleExample ? columnStandardDeviations : ones.mult(columnStandardDeviations);
		return data.minus(means).elementDiv(standardDeviations);
	}
}
