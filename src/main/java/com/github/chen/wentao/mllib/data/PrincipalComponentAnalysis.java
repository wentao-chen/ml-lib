package com.github.chen.wentao.mllib.data;

import com.github.chen.wentao.mllib.training.DataSet;
import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleSVD;

public class PrincipalComponentAnalysis {

	public static int findOptimalTargetFeaturesCount(DataSet data) {
		return findOptimalTargetFeaturesCount(data, 0.99);
	}

	public static int findOptimalTargetFeaturesCount(DataSet data, double minimumVarianceRetained) {
		return findOptimalTargetFeaturesCount(data.getMatrix(), minimumVarianceRetained);
	}

	private static int findOptimalTargetFeaturesCount(SimpleMatrix data, double minimumVarianceRetained) {
		SimpleSVD<SimpleMatrix> svd = svd(data);
		SimpleMatrix svdSingularValues = svd.getW();
		int n = data.numCols();
		double svdSingularValuesSum = svdSingularValues.elementSum();
		double runningSum = 0.0;
		for (int k = 0; k < n - 1; k++) {
			runningSum += svdSingularValues.get(k, k);
			double varianceRetained = runningSum / svdSingularValuesSum;
			if (varianceRetained >= minimumVarianceRetained) {
				return k + 1;
			}
		}
		return n;
	}

	public static double cost(DataSet dataSet, int targetFeaturesCount) {
		return cost(dataSet.getMatrix(), targetFeaturesCount);
	}

	private static double cost(SimpleMatrix data, int targetFeaturesCount) {
		assert targetFeaturesCount > 0;
		if (targetFeaturesCount >= data.numCols()) return 0.0;

		SimpleMatrix svdUMatrix = svd(data).getU();
		SimpleMatrix dataApprox = expand(reduce(data, svdUMatrix, targetFeaturesCount), svdUMatrix);
		double cost = 0.0;
		for (int i = 0, m = data.numRows(); i < m; i++) {
			cost += data.rows(i, i + 1).minus(dataApprox.rows(i, i + 1)).elementPower(2.0).elementSum();
		}
		return cost / data.numRows();
	}

	public static double dataVariation(DataSet dataSet) {
		return dataVariation(dataSet.getMatrix());
	}

	private static double dataVariation(SimpleMatrix data) {
		double sum = 0.0;
		for (int i = 0, m = data.numRows(); i < m; i++) {
			sum += data.rows(i, i + 1).elementPower(2.0).elementSum();
		}
		return sum / data.numRows();
	}

	public static double varianceRetained(DataSet dataSet, int targetFeaturesCount) {
		return varianceRetained(dataSet.getMatrix(), targetFeaturesCount);
	}

	private static double varianceRetained(SimpleMatrix data, int targetFeaturesCount) {
		return 1.0 - cost(data, targetFeaturesCount) / dataVariation(data);
	}

	public static DataSet reduce(DataSet data, int targetFeaturesCount) {
		return new DataSet(reduce(data.getMatrix(), targetFeaturesCount));
	}

	private static SimpleMatrix reduce(SimpleMatrix data, int targetFeaturesCount) {
		assert targetFeaturesCount > 0;
		if (targetFeaturesCount >= data.numCols()) {
			return data;
		}
		return reduce(data, svd(data).getU(), targetFeaturesCount);
	}

	private static SimpleMatrix reduce(SimpleMatrix data, SimpleMatrix svdUMatrix, int targetFeaturesCount) {
		return data.mult(svdUMatrix.cols(0, targetFeaturesCount));
	}

	private static SimpleMatrix expand(SimpleMatrix data, SimpleMatrix svdUMatrix) {
		return data.mult(svdUMatrix.cols(0, data.numCols()).transpose());
	}

	private static SimpleSVD<SimpleMatrix> svd(SimpleMatrix data) {
		double m = data.numRows();
		SimpleMatrix covarianceMatrix = data.transpose().mult(data).divide(m);
		return covarianceMatrix.svd();
	}
}
