package com.github.chen.wentao.mllib.data;

import com.github.chen.wentao.mllib.training.DataSet;
import com.github.chen.wentao.mllib.training.DataSetTarget;
import com.github.chen.wentao.mllib.util.ejml.SimpleMatrixUtil;
import org.ejml.data.MatrixType;
import org.ejml.simple.SimpleMatrix;

import java.util.function.DoubleFunction;

import static com.github.chen.wentao.mllib.util.ejml.SimpleMatrixUtil.ones;

public class AnomalyDetector {

	private static final double TAU = 2.0 * Math.PI;
	private static final double TAU_ROOT = Math.sqrt(TAU);

	public static double findOptimalEpsilon(DataSet trainingDataSet, DataSet validationSet, DataSetTarget validationSetTarget, double[] epsilonTestValues) {
		return findOptimalEpsilon(epsilon -> findAnomalies(trainingDataSet.getMatrix(), epsilon), validationSet.getMatrix(), validationSetTarget.getMatrix(), epsilonTestValues);
	}

	public static double findOptimalEpsilonMultiVariate(DataSet trainingDataSet, DataSet validationSet, DataSetTarget validationSetTarget, double[] epsilonTestValues) {
		return findOptimalEpsilon(epsilon -> findAnomaliesMultiVariate(trainingDataSet.getMatrix(), epsilon), validationSet.getMatrix(), validationSetTarget.getMatrix(), epsilonTestValues);
	}

	private static double findOptimalEpsilon(DoubleFunction<SimpleMatrix> predictor, SimpleMatrix validationSet, SimpleMatrix validationSetTarget, double[] epsilonTestValues) {
		assert validationSet.numRows() == validationSetTarget.numRows(); // Correct number of validation examples
		assert validationSetTarget.numCols() == 1;

		int validationM = validationSet.numRows();
		double bestEpsilon = Double.NaN;
		double bestEpsilonF1Score = Double.NEGATIVE_INFINITY;
		for (double epsilon : epsilonTestValues) {
			SimpleMatrix predictions = predictor.apply(epsilon);
			int truePositives = 0;
			int falsePositives = 0;
			int falseNegatives = 0;
			for (int i = 0; i < validationM; i++) {
				double predictedValue = predictions.get(i);
				double actualValue = validationSetTarget.get(i);
				if (predictedValue != 0 && actualValue != 0) {
					truePositives += 1;
				} else if (predictedValue != 0) {
					falsePositives += 1;
				} else if (actualValue != 0) {
					falseNegatives += 1;
				}
			}
			double precision = truePositives / (truePositives + falsePositives);
			double recall = truePositives / (truePositives + falseNegatives);
			double f1Score = 2.0 * precision * recall / (precision + recall);
			if (f1Score > bestEpsilonF1Score) {
				bestEpsilon = epsilon;
				bestEpsilonF1Score = f1Score;
			}
		}
		return bestEpsilon;
	}

	public static boolean isAnomaly(double x, double mean, double variance, double epsilon) {
		return gaussian(x, mean, variance) < epsilon;
	}

	public static DataSetTarget findAnomalies(DataSet dataSet, double epsilon) {
		return new DataSetTarget(findAnomalies(dataSet.getMatrix(), epsilon), 2);
	}

	private static SimpleMatrix findAnomalies(SimpleMatrix data, double epsilon) {
		return findAnomalies(gaussian(data), data, epsilon);
	}

	public static DataSetTarget findAnomaliesMultiVariate(DataSet dataSet, double epsilon) {
		return new DataSetTarget(findAnomaliesMultiVariate(dataSet.getMatrix(), epsilon), 2);
	}

	private static SimpleMatrix findAnomaliesMultiVariate(SimpleMatrix data, double epsilon) {
		return findAnomalies(gaussianMultiVariate(data), data, epsilon);
	}

	private static SimpleMatrix findAnomalies(SimpleMatrix gaussian, SimpleMatrix data, double epsilon) {
		SimpleMatrix anomalies = new SimpleMatrix(data.numRows(), 1, MatrixType.DDRM);
		int n = data.numCols();
		for (int i = 0, m = data.numRows(); i < m; i++) {
			double totalProbability = 1.0;
			for (int x = 0; i < n; i++) {
				totalProbability *= gaussian.get(i, x);
			}
			anomalies.set(i, totalProbability < epsilon ? 1.0 : 0.0);
		}
		return anomalies;
	}

	public static double gaussian(double x, double mean, double variance) {
		double diff = x - mean;
		return Math.exp(- diff * diff / 2.0 / variance) / Math.sqrt(TAU * variance);
	}

	public static DataSet gaussian(DataSet data) {
		return new DataSet(gaussian(data.getMatrix()));
	}

	private static SimpleMatrix gaussian(SimpleMatrix data) {
		SimpleMatrix ones = ones(data.numRows(), 1);
		SimpleMatrix mean = ones.mult(dataMeans(data));
		SimpleMatrix variance = ones.mult(dataVariances(data));
		return data.minus(mean).elementPower(2.0).divide(2.0).elementDiv(variance).negative().elementExp().divide(TAU_ROOT).elementDiv(variance.elementPower(0.5));
	}

	public static DataSet gaussianMultiVariate(DataSet data) {
		return new DataSet(gaussianMultiVariate(data.getMatrix()));
	}

	private static SimpleMatrix gaussianMultiVariate(SimpleMatrix data) {
		SimpleMatrix ones = ones(data.numRows(), 1);
		SimpleMatrix mean = ones.mult(dataMeans(data));
		SimpleMatrix covarianceMatrix = data.transpose().mult(data).divide(data.numRows());
		SimpleMatrix diff = data.minus(mean);
		return diff.transpose().mult(covarianceMatrix.pseudoInverse()).mult(diff).divide(-2.0).elementExp().divide(Math.pow(TAU, data.numCols() / 2.0) * Math.sqrt(covarianceMatrix.determinant()));
	}

	private static SimpleMatrix dataMeans(SimpleMatrix data) {
		return SimpleMatrixUtil.sumCols(data).divide(data.numRows());
	}

	private static SimpleMatrix dataVariances(SimpleMatrix data) {
		return SimpleMatrixUtil.sumCols(data.elementPower(2.0)).divide(data.numRows()).minus(dataMeans(data).elementPower(2.0));
	}
}
