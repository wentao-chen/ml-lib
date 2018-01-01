package com.github.chen.wentao.mllib.training;

@FunctionalInterface
public interface AlgorithmHypothesis {

	DataSetTarget apply(DataSet dataSet);

	default double findAccuracy(FullDataSet dataSet) {
		DataSetTarget predicted = apply(dataSet.getDataSet());
		DataSetTarget actual = dataSet.getDataSetTarget();

		int correctPredictions = 0;
		int m = dataSet.numExamples();
		for (int i = 0; i < m; i++) {
			if (predicted.get(i) == actual.get(i)) {
				correctPredictions += 1;
			}
		}
		return (double) correctPredictions / (double) m;
	}

	default double findF1Score(FullDataSet dataSet) {
		DataSetTarget predicted = apply(dataSet.getDataSet());
		DataSetTarget actual = dataSet.getDataSetTarget();

		int truePositives = 0;
		int falsePositives = 0;
		int falseNegatives = 0;
		int m = dataSet.numExamples();
		for (int i = 0; i < m; i++) {
			double predictedValue = predicted.get(i);
			double actualValue = actual.get(i);
			if (predictedValue != 0 && actualValue != 0) {
				truePositives += 1;
			} else if (predictedValue != 0) {
				falsePositives += 1;
			} else if (actualValue != 0) {
				falseNegatives += 1;
			}
		}
		double precision = (double) truePositives / (truePositives + falsePositives);
		double recall = (double) truePositives / (truePositives + falseNegatives);
		return 2.0 * precision * recall / (precision + recall);
	}
}
