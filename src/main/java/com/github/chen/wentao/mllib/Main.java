package com.github.chen.wentao.mllib;

import com.github.chen.wentao.mllib.data.scaling.FeatureScaler;
import com.github.chen.wentao.mllib.data.scaling.FeatureStandardizer;
import com.github.chen.wentao.mllib.training.*;
import org.ejml.simple.SimpleMatrix;

public class Main {

	public static void main(String[] args) {
		logisticRegressionTest();
	}

	private static void logisticRegressionTest() {
		DataSet dataSet = new DataSet(new SimpleMatrix(29 + 9, 2, true, 1.0, 2.0, 4.0, 6.0, 3.0, 4.0, 8.0, 0.0, 7.5, 5.0, 2.0, 0.3, 4.0, 1.0, 6.0, 0.1, 2.0, 7.0, 7.0, 1.0, 1.7, 3.0, 3.0, 0.8, 6.0, 5.2, 7.8, 3.9, 6.9, 2.5, 5.0, 5.0, 4.0, 3.0, 5.0, 4.0, 6.0, 3.0, 3.0, 3.0, 5.0, 2.0, 4.2, 2.1, 4.3, 1.8, 5.8, 2.8, 4.3, 3.9, 5.2, 3.5, 4.8, 2.7, 6.5, 4.2, 5.7, 1.7, 8, 6, 7, 7, 7, 6, 8, 8, 8, 7, 7, 8, 9, 8, 9, 7, 9, 6)).addPowerTerms(2.0);
		LabeledDataSetTarget target = new LabeledDataSetTarget(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2);
		FeatureParameters initial = new FeatureParameters(0.0, 0.0, 0.0, 0.0, 0.0);

		FeatureScaler featureScaler = new FeatureStandardizer(dataSet);
		DataSet normalizedDataSet = featureScaler.getNormalizedDataSet();

		long start = System.nanoTime();
		FeatureParameters[] thetas = LogisticRegression.gradientDescentMulti(normalizedDataSet, target, initial, 3, 1.0, 100000);
		long timeElapsed = System.nanoTime() - start;

		System.out.printf("Time: %.6fms%n", timeElapsed / 1000000.0);
		System.out.printf("Final Cost: %.6f%n", LogisticRegression.costFunctionMulti(thetas, normalizedDataSet, target, 1.0));
		System.out.printf("%d%n", LogisticRegression.predictMulti(thetas, featureScaler.normalize(DataSet.single(7.43, 7.51).addPowerTerms(2.0))).getLabel(0));
	}

	private static void linearRegressionTest() {
		DataSet dataSet = new DataSet(new SimpleMatrix(10, 1, true, 75.0, 3.0, 21.0, 13.0, 10.0, 13.0, 40.0, 30.0, 60.0, 50.0)).addPowerTerms(2.0, 3.0);
		DataSetTarget target = new DataSetTarget(70.0, 12.0, 50.0, 65.0, 35.0, 54.0, 10.0, 40.0, 60.0, 20.0);
		FeatureParameters initial = new FeatureParameters(0.0, 0.0, 0.0, 0.0);

		FeatureScaler featureScaler = new FeatureStandardizer(dataSet);
		DataSet normalizedDataSet = featureScaler.getNormalizedDataSet();

		long start = System.nanoTime();
		FeatureParameters theta = LinearRegression.gradientDescent(normalizedDataSet, target, initial, 0.1, 1.0, 100000);
		long timeElapsed = System.nanoTime() - start;

		theta.print(15, 10);
		System.out.printf("Time: %.6fms%n", timeElapsed / 1000000.0);
		System.out.printf("Final Cost: %.6f%n", LinearRegression.costFunction(theta, normalizedDataSet, target, 1.0));


		// Normal equation

		long start2 = System.nanoTime();
		FeatureParameters thetaNormal = LinearRegression.normalEquation(normalizedDataSet, target, 1.0);
		long timeElapsed2 = System.nanoTime() - start2;

		thetaNormal.print(15, 10);
		System.out.printf("Time: %.6fms%n", timeElapsed2 / 1000000.0);
		System.out.printf("Final Cost: %.6f%n", LinearRegression.costFunction(thetaNormal, normalizedDataSet, target, 1.0));
	}
}
