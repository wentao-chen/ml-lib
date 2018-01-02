package com.github.chen.wentao.mllib;

import com.github.chen.wentao.mllib.data.LearningCurve;
import com.github.chen.wentao.mllib.data.scaling.FeatureScaler;
import com.github.chen.wentao.mllib.data.scaling.FeatureStandardizer;
import com.github.chen.wentao.mllib.training.*;
import org.ejml.simple.SimpleMatrix;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.util.Map;
import java.util.Random;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class Main {

	public static void main(String[] args) {
		kMeansClusteringTest();
	}

	private static void kMeansClusteringTest() {
		DataSet dataSet = new DataSet.Builder()
				.add(1.0, 1.0)
				.add(1.0, 2.0)
				.add(2.0, 1.0)
				.add(6.0, 7.0)
				.add(7.0, 7.0)
				.add(7.0, 6.0)
				.add(13.0, 14.0)
				.add(14.0, 14.0)
				.add(14.0, 13.0)
				.build();

		Map<Integer, Double> kMeansCosts = KMeansClustering.kMeansCostPerCluster(dataSet, new Random(), 10, IntStream.range(1, dataSet.numExamples()).toArray());
		XYSeries series = new XYSeries("Data");
		kMeansCosts.forEach(series::add);

		JFreeChart chart = ChartFactory.createXYLineChart("K-Means Clustering", "Number of clusters (K)", "Cost", new XYSeriesCollection(series), PlotOrientation.VERTICAL, true, true, true);
		JFrame frame = new JFrame("K-Means Clustering");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.add(new ChartPanel(chart));
		frame.setSize(600, 600);
		frame.setVisible(true);
	}

	private static void findingOptimalLambdaTest() {
		FullDataSet fullDataSet = new FullDataSet.Builder()
				.add(1.0, 2.0, 0).add(4.0, 3.0, 1).add(8.0, 6.0, 2)
				.add(4.0, 6.0, 0).add(5.0, 4.0, 1).add(7.0, 7.0, 2)
				.add(3.0, 4.0, 0).add(6.0, 3.0, 1).add(7.0, 6.0, 2)
				.add(8.0, 0.0, 0).add(3.0, 3.0, 1).add(8.0, 8.0, 2)
				.add(7.5, 5.0, 0).add(5.0, 2.0, 1).add(8.0, 7.0, 2)
				.add(2.0, 0.3, 0).add(4.2, 2.1, 1).add(7.0, 8.0, 2)
				.add(4.0, 1.0, 0).add(4.3, 1.8, 1).add(9.0, 8.0, 2)
				.add(6.0, 0.1, 0).add(5.8, 2.8, 1).add(9.0, 7.0, 2)
				.add(2.0, 7.0, 0).add(4.3, 3.9, 1).add(9.0, 6.0, 2)
				.add(7.0, 1.0, 0).add(5.2, 3.5, 1)
				.add(1.7, 3.0, 0).add(4.8, 2.7, 1)
				.add(3.0, 0.8, 0).add(6.5, 4.2, 1)
				.add(6.0, 5.2, 0).add(5.7, 1.7, 1)
				.add(7.8, 3.9, 0)
				.add(6.9, 2.5, 0)
				.add(5.0, 5.0, 0)
				.addPowerTerms(2.0)
				.buildAndNormalize();
		FeatureParameters initial = new FeatureParameters(0.0, 0.0, 0.0, 0.0, 0.0);

		TrainCVTestDataSet trainCVTestDataSet = TrainCVTestDataSet.fromFullDataSet(fullDataSet, new Random());

		DoubleStream testLambdaValues = DoubleStream.of(0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24);

		LearningCurve.generateLearningCurve(
				lambda -> LogisticRegression.getAlgorithmMulti(initial, 0.1, lambda, 10000),
				LogisticRegression::getCostFunctionMulti,
				trainCVTestDataSet,
				testLambdaValues
		).graphWithJFrame("Regularization Parameter", "Lambda", false, 600, 600);

		FeatureParameters[] optimal = LogisticRegression.gradientDescentMulti(fullDataSet.getDataSet(), fullDataSet.getDataSetTarget(), initial, 0.1, 0.0, 10000);
		System.out.println(LogisticRegression.getMultiPredictorHypothesis(optimal).findAccuracy(fullDataSet));
		System.out.println(LogisticRegression.getMultiPredictorHypothesis(optimal).findF1Score(fullDataSet));
	}

	private static void learningCurvesTest() {
		FullDataSet fullDataSet = new FullDataSet.Builder()
				.add(1.0, 2.0, 0).add(4.0, 3.0, 1).add(8.0, 6.0, 2)
				.add(4.0, 6.0, 0).add(5.0, 4.0, 1).add(7.0, 7.0, 2)
				.add(3.0, 4.0, 0).add(6.0, 3.0, 1).add(7.0, 6.0, 2)
				.add(8.0, 0.0, 0).add(3.0, 3.0, 1).add(8.0, 8.0, 2)
				.add(7.5, 5.0, 0).add(5.0, 2.0, 1).add(8.0, 7.0, 2)
				.add(2.0, 0.3, 0).add(4.2, 2.1, 1).add(7.0, 8.0, 2)
				.add(4.0, 1.0, 0).add(4.3, 1.8, 1).add(9.0, 8.0, 2)
				.add(6.0, 0.1, 0).add(5.8, 2.8, 1).add(9.0, 7.0, 2)
				.add(2.0, 7.0, 0).add(4.3, 3.9, 1).add(9.0, 6.0, 2)
				.add(7.0, 1.0, 0).add(5.2, 3.5, 1)
				.add(1.7, 3.0, 0).add(4.8, 2.7, 1)
				.add(3.0, 0.8, 0).add(6.5, 4.2, 1)
				.add(6.0, 5.2, 0).add(5.7, 1.7, 1)
				.add(7.8, 3.9, 0)
				.add(6.9, 2.5, 0)
				.add(5.0, 5.0, 0)
				.addPowerTerms(2.0)
				.buildAndNormalize();
		FeatureParameters initial = new FeatureParameters(0.0, 0.0, 0.0, 0.0, 0.0);
		double lambda = 0.5;

		SupervisedLearningAlgorithm<FeatureParameters[]> algorithm = LogisticRegression.getAlgorithmMulti(initial, 0.1, lambda, 100000);
		CostFunction<FeatureParameters[]> costFunction = LogisticRegression.getCostFunctionMulti(lambda);
		TrainCVTestDataSet trainCVTestDataSet = TrainCVTestDataSet.fromFullDataSet(fullDataSet, new Random());

		LearningCurve.generateSetSizeLearningCurve(algorithm, costFunction, trainCVTestDataSet.getFullTrainingSet(), trainCVTestDataSet.getFullCrossValidationSet(), new Random(), 10)
				.graphWithJFrame("Learning Curves", "Number of training examples", false, 600, 600);
	}

	private static void neuralNetworkTest() {
		NeuralNetwork neuralNetwork = NeuralNetwork.emptyNetwork(2, 2, 1).randomlyInitialize(new Random());
		FullDataSet fullDataSet = new FullDataSet.Builder()
				.add(1.0, 1.0, 0)
				.add(1.0, 0.0, 1)
				.add(0.0, 1.0, 1)
				.add(0.0, 0.0, 0)
				.build();

		fullDataSet.print(15, 10);
		fullDataSet = fullDataSet.shuffle(new Random());
		fullDataSet.print(15, 10);

		System.out.println(neuralNetwork.costFunction(fullDataSet.getDataSet(), fullDataSet.getDataSetTarget(), 0.0));

		neuralNetwork.train(fullDataSet.getDataSet(), fullDataSet.getDataSetTarget(), 1.0, 0.0, 50000);

		System.out.println(neuralNetwork.costFunction(fullDataSet.getDataSet(), fullDataSet.getDataSetTarget(), 0.0));

		neuralNetwork.predict(DataSet.single(1.0, 1.0)).print(15, 10);
		neuralNetwork.predict(DataSet.single(0.0, 1.0)).print(15, 10);
		neuralNetwork.predict(DataSet.single(1.0, 0.0)).print(15, 10);
		neuralNetwork.predict(DataSet.single(0.0, 0.0)).print(15, 10);

		neuralNetwork.print(15, 10);
	}

	private static void logisticRegressionTest() {
		DataSet dataSet = new DataSet(new SimpleMatrix(29 + 9, 2, true, 1.0, 2.0, 4.0, 6.0, 3.0, 4.0, 8.0, 0.0, 7.5, 5.0, 2.0, 0.3, 4.0, 1.0, 6.0, 0.1, 2.0, 7.0, 7.0, 1.0, 1.7, 3.0, 3.0, 0.8, 6.0, 5.2, 7.8, 3.9, 6.9, 2.5, 5.0, 5.0, 4.0, 3.0, 5.0, 4.0, 6.0, 3.0, 3.0, 3.0, 5.0, 2.0, 4.2, 2.1, 4.3, 1.8, 5.8, 2.8, 4.3, 3.9, 5.2, 3.5, 4.8, 2.7, 6.5, 4.2, 5.7, 1.7, 8, 6, 7, 7, 7, 6, 8, 8, 8, 7, 7, 8, 9, 8, 9, 7, 9, 6)).addPowerTerms(2.0);
		DataSetTarget target = new DataSetTarget(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2);
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
