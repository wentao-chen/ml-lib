package com.github.chen.wentao.mllib.data;

import com.github.chen.wentao.mllib.training.*;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.stream.IntStream;

public class LearningCurve {
	private Map<Integer, Double> trainError = new HashMap<>();
	private Map<Integer, Double> cvError = new HashMap<>();

	private LearningCurve() {
	}

	private void addTrainError(int size, double error) {
		if (Double.isFinite(error)) {
			trainError.put(size, error);
		}
	}

	private void addCVError(int size, double error) {
		if (Double.isFinite(error)) {
			cvError.put(size, error);
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

	public JFreeChart graphToChart() {
		XYSeries trainSeries = new XYSeries("Training Set");
		trainError.forEach(trainSeries::add);
		XYSeries cvSeries = new XYSeries("Cross Validation Set");
		cvError.forEach(cvSeries::add);

		XYSeriesCollection xyDataSet = new XYSeriesCollection();
		xyDataSet.addSeries(trainSeries);
		xyDataSet.addSeries(cvSeries);
		return ChartFactory.createXYLineChart("Learning Curves", "Number of training examples", "Error", xyDataSet, PlotOrientation.VERTICAL, true, true, true);
	}

	public void graphWithJFrame(int width, int height) {
		JFreeChart chart = graphToChart();
		JFrame frame = new JFrame("Learning Curves");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		JPanel chartPanel = new ChartPanel(chart);
		frame.add(chartPanel);
		frame.setSize(width, height);
		frame.setVisible(true);
		chartPanel.revalidate();
	}
}
