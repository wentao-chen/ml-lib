package com.github.chen.wentao.mllib.data;

import com.github.chen.wentao.mllib.training.*;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.LogAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.BorderLayout;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.DoubleFunction;
import java.util.function.IntFunction;
import java.util.logging.Logger;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class LearningCurve {

	private static final Logger LOGGER = Logger.getLogger(LearningCurve.class.getName());

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

	public Map<Double, Double> getTrainingError() {
		return new HashMap<>(trainError);
	}

	public Map<Double, Double> getCrossValidationError() {
		return new HashMap<>(cvError);
	}

	public Map<Double, Double> getTestError() {
		return new HashMap<>(testError);
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

	public static <T> LearningCurve generateSetSizeLearningCurve(StreamSupervisedLearningAlgorithm<T> learningAlgorithm, StreamCostFunction<T> cost, IntFunction<BatchFullDataSetStream> trainingDataSetGenerator, BatchFullDataSetStream crossValidationDataSet, int[] testSizes) {
		LearningCurve learningCurve = new LearningCurve();
		for (int size : testSizes) {
			LOGGER.info(() -> "Generating learning curve for test size " + size);
			BatchFullDataSetStream trainDataSubset = trainingDataSetGenerator.apply(size);
			LOGGER.info(() -> "Training network for test size " + size);
			T optimalParamsTrain = learningAlgorithm.findOptimalParameters(trainDataSubset);
			LOGGER.info(() -> "Calculating learning curve costs for test size " + size);
			learningCurve.addTrainError(size, cost.apply(optimalParamsTrain, trainDataSubset));
			learningCurve.addCVError(size, cost.apply(optimalParamsTrain, crossValidationDataSet));
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
		frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
		JPanel mainPanel = new JPanel(new BorderLayout());
		mainPanel.add(new ChartPanel(chart), BorderLayout.CENTER);
		JPanel buttonPanel = new JPanel(new BorderLayout());
		JButton saveButton = new JButton("Save");
		saveButton.addActionListener(e -> {
			JFileChooser fileChooser = new JFileChooser(System.getProperty("user.dir"));
			fileChooser.setFileFilter(new FileNameExtensionFilter(".png", "png"));
			if (fileChooser.showSaveDialog(frame) == JFileChooser.APPROVE_OPTION) {
				String fileName = fileChooser.getSelectedFile().toString();
				if (!fileName.toLowerCase().endsWith(".png")) {
					fileName += ".png";
				}
				File file = new File(fileName);
				if (file.exists()) {
					if (JOptionPane.showConfirmDialog(frame, "Overwrite existing: " + fileName) != JOptionPane.YES_OPTION) {
						return;
					}
				}
				try {
					ChartUtilities.saveChartAsPNG(file, chart, width, height);
					JOptionPane.showMessageDialog(frame, "Saved");
				} catch (IOException e1) {
					JOptionPane.showMessageDialog(frame, "IOException " + e1);
				}
			}
		});
		buttonPanel.add(saveButton, BorderLayout.EAST);
		mainPanel.add(buttonPanel, BorderLayout.SOUTH);
		frame.add(mainPanel);
		frame.setSize(width, height);
		frame.setVisible(true);
	}
}
