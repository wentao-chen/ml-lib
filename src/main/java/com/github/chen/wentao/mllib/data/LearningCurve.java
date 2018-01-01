package com.github.chen.wentao.mllib.data;

import com.github.chen.wentao.mllib.training.*;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.LogAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.function.DoubleFunction;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class LearningCurve {

	private final Map<Double, Double> trainError = new HashMap<>();
	private final Map<Double, Double> cvError = new HashMap<>();
	private final Map<Double, Double> testError = new HashMap<>();

	private LearningCurve() {
	}

	private void addTrainError(double key, double error) {
		if (Double.isFinite(key) && Double.isFinite(error)) {
			trainError.put(key, error);
		}
	}

	private void addCVError(double key, double error) {
		if (Double.isFinite(key) && Double.isFinite(error)) {
			cvError.put(key, error);
		}
	}

	private void addTestError(double key, double error) {
		if (Double.isFinite(key) && Double.isFinite(error)) {
			testError.put(key, error);
		}
	}

	public static <T> LearningCurve generateSetSizeLearningCurve(SupervisedLearningAlgorithm<T> learningAlgorithm, CostFunction<T> cost, FullDataSet trainingDataSet, FullDataSet crossValidationDataSet, Random random) {
		return generateSetSizeLearningCurve(learningAlgorithm, cost, trainingDataSet, crossValidationDataSet, random, IntStream.range(1, trainingDataSet.numExamples() + 1).toArray());
	}

	public static <T> LearningCurve generateSetSizeLearningCurve(SupervisedLearningAlgorithm<T> learningAlgorithm, CostFunction<T> cost, FullDataSet trainingDataSet, FullDataSet crossValidationDataSet, Random random, int testCount) {
		int m = trainingDataSet.numExamples();
		return generateSetSizeLearningCurve(learningAlgorithm, cost, trainingDataSet, crossValidationDataSet, random, IntStream.range(0, testCount).map(i -> Math.min(i * m / testCount + 1, m)).toArray());
	}

	public static <T> LearningCurve generateSetSizeLearningCurve(SupervisedLearningAlgorithm<T> learningAlgorithm, CostFunction<T> cost, FullDataSet trainingDataSet, FullDataSet crossValidationDataSet, Random random, int[] testSizes) {
		DataSet cvDataSet = crossValidationDataSet.getDataSet();
		DataSetTarget cvDataSetTarget = crossValidationDataSet.getDataSetTarget();
		LearningCurve learningCurve = new LearningCurve();
		for (int size : testSizes) {
			FullDataSet trainDataSubset = trainingDataSet.shuffle(random).subset(size);
			T optimalParamsTrain = learningAlgorithm.findOptimalParameters(trainDataSubset.getDataSet(), trainDataSubset.getDataSetTarget());
			learningCurve.addTrainError(size, cost.apply(optimalParamsTrain, trainDataSubset.getDataSet(), trainDataSubset.getDataSetTarget()));
			learningCurve.addCVError(size, cost.apply(optimalParamsTrain, cvDataSet, cvDataSetTarget));
		}
		return learningCurve;
	}

	public static <T> LearningCurve generateLearningCurve(DoubleFunction<SupervisedLearningAlgorithm<T>> learningAlgorithmGenerator, DoubleFunction<CostFunction<T>> costFunctionGenerator, DoubleFunction<FullDataSet> trainingDataSetGenerator, FullDataSet crossValidationDataSet, DoubleStream testValues) {
		DataSet cvDataSet = crossValidationDataSet.getDataSet();
		DataSetTarget cvDataSetTarget = crossValidationDataSet.getDataSetTarget();
		LearningCurve learningCurve = new LearningCurve();
		testValues.forEach(value -> {
			FullDataSet trainDataSubset = trainingDataSetGenerator.apply(value);
			SupervisedLearningAlgorithm<T> algorithm = learningAlgorithmGenerator.apply(value);
			CostFunction<T> costFunction = costFunctionGenerator.apply(value);
			T optimalParamsTrain = algorithm.findOptimalParameters(trainDataSubset.getDataSet(), trainDataSubset.getDataSetTarget());
			learningCurve.addTrainError(value, costFunction.apply(optimalParamsTrain, trainDataSubset.getDataSet(), trainDataSubset.getDataSetTarget()));
			learningCurve.addCVError(value, costFunction.apply(optimalParamsTrain, cvDataSet, cvDataSetTarget));
		});
		return learningCurve;
	}

	public static <T> LearningCurve generateLearningCurve(DoubleFunction<SupervisedLearningAlgorithm<T>> learningAlgorithmGenerator, DoubleFunction<CostFunction<T>> costFunctionGenerator, TrainCVTestDataSet fullDataSet, DoubleStream testValues) {
		DataSet trainingDataSet = fullDataSet.getTrainingSet();
		DataSetTarget trainingDataSetTarget = fullDataSet.getTrainingSetTarget();
		DataSet cvDataSet = fullDataSet.getCrossValidationSet();
		DataSetTarget cvDataSetTarget = fullDataSet.getCrossValidationSetTarget();
		DataSet testDataSet = fullDataSet.getTestSet();
		DataSetTarget testDataSetTarget = fullDataSet.getTestSetTarget();
		LearningCurve learningCurve = new LearningCurve();
		testValues.forEach(value -> {
			SupervisedLearningAlgorithm<T> algorithm = learningAlgorithmGenerator.apply(value);
			CostFunction<T> costFunction = costFunctionGenerator.apply(value);
			T optimalParamsTrain = algorithm.findOptimalParameters(trainingDataSet, trainingDataSetTarget);
			learningCurve.addTrainError(value, costFunction.apply(optimalParamsTrain, trainingDataSet, trainingDataSetTarget));
			learningCurve.addCVError(value, costFunction.apply(optimalParamsTrain, cvDataSet, cvDataSetTarget));
			learningCurve.addTestError(value, costFunction.apply(optimalParamsTrain, testDataSet, testDataSetTarget));
		});
		return learningCurve;
	}

	public JFreeChart graphToChart(String title, String xAxisLabel, boolean useLogAxis) {
		XYSeries trainSeries = new XYSeries("Training Set");
		trainError.forEach(trainSeries::add);
		XYSeries cvSeries = new XYSeries("Cross Validation Set");
		cvError.forEach(cvSeries::add);
		XYSeries testSeries = new XYSeries("Test Set");
		testError.forEach(testSeries::add);

		XYSeriesCollection xyDataSet = new XYSeriesCollection();
		if (!trainError.isEmpty()) xyDataSet.addSeries(trainSeries);
		if (!cvError.isEmpty()) xyDataSet.addSeries(cvSeries);
		if (!testError.isEmpty()) xyDataSet.addSeries(testSeries);
		JFreeChart chart = ChartFactory.createXYLineChart(title, xAxisLabel, "Error", xyDataSet, PlotOrientation.VERTICAL, true, true, true);
		if (useLogAxis) chart.getXYPlot().setDomainAxis(new LogAxis(xAxisLabel));
		return chart;
	}

	public void graphWithJFrame(String title, String xAxisLabel, boolean useLogAxis, int width, int height) {
		JFreeChart chart = graphToChart(title, xAxisLabel, useLogAxis);
		JFrame frame = new JFrame("Learning Curves");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		JPanel chartPanel = new ChartPanel(chart);
		frame.add(chartPanel);
		frame.setSize(width, height);
		frame.setVisible(true);
		chartPanel.revalidate();
	}
}
