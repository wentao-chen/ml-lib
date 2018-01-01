package com.github.chen.wentao.mllib.training;

import org.ejml.simple.SimpleMatrix;

import java.util.Random;

public class TrainCVTestDataSet {

	private final DataSet trainingSet;
	private final DataSetTarget trainingSetTarget;
	private final DataSet crossValidationSet;
	private final DataSetTarget crossValidationSetTarget;
	private final DataSet testSet;
	private final DataSetTarget testSetTarget;
	private final FullDataSet fullTrainingSet;
	private final FullDataSet fullCrossValidationSet;
	private final FullDataSet fullTestSet;

	public TrainCVTestDataSet(DataSet trainingSet, DataSetTarget trainingSetTarget, DataSet crossValidationSet, DataSetTarget crossValidationSetTarget, DataSet testSet, DataSetTarget testSetTarget) {
		this.trainingSet = trainingSet;
		this.trainingSetTarget = trainingSetTarget;
		this.crossValidationSet = crossValidationSet;
		this.crossValidationSetTarget = crossValidationSetTarget;
		this.testSet = testSet;
		this.testSetTarget = testSetTarget;
		this.fullTrainingSet = new FullDataSet(trainingSet, trainingSetTarget);
		this.fullCrossValidationSet = new FullDataSet(crossValidationSet, crossValidationSetTarget);
		this.fullTestSet = new FullDataSet(testSet, testSetTarget);
	}

	public static TrainCVTestDataSet fromFullDataSet(FullDataSet fullDataSet,  Random random) {
		return fromFullDataSet(fullDataSet, random, 0.6, 0.2, 0.2);
	}

	public static TrainCVTestDataSet fromFullDataSet(FullDataSet fullDataSet, Random random, double trainProportion, double cvProportion, double testProportion) {
		fullDataSet = fullDataSet.shuffle(random);

		DataSet dataSet = fullDataSet.getDataSet();
		DataSetTarget target = fullDataSet.getDataSetTarget();

		// Normalize proportions
		double totalProportion = trainProportion + cvProportion + testProportion;
		cvProportion /= totalProportion;
		testProportion /= totalProportion;

		int m = dataSet.numExamples();
		int testSize = (int) (testProportion * m);
		int cvSize = (int) (cvProportion * m);
		int trainSize = m - cvSize - testSize;

		SimpleMatrix dataSetMatrix = dataSet.getMatrix();
		SimpleMatrix targetMatrix = target.getMatrix();

		DataSet train = new DataSet(dataSetMatrix.extractMatrix(0, trainSize, 0, dataSetMatrix.numCols()));
		DataSetTarget trainTarget = new DataSetTarget(targetMatrix.extractMatrix(0, trainSize, 0, targetMatrix.numCols()), target.numLabels());
		DataSet cv = new DataSet(dataSetMatrix.extractMatrix(trainSize, trainSize + cvSize, 0, dataSetMatrix.numCols()));
		DataSetTarget cvTarget = new DataSetTarget(targetMatrix.extractMatrix(trainSize, trainSize + cvSize, 0, targetMatrix.numCols()), target.numLabels());
		DataSet test = new DataSet(dataSetMatrix.extractMatrix(trainSize + cvSize, dataSetMatrix.numRows(), 0, dataSetMatrix.numCols()));
		DataSetTarget testTarget = new DataSetTarget(targetMatrix.extractMatrix(trainSize + cvSize, targetMatrix.numRows(), 0, targetMatrix.numCols()), target.numLabels());
		return new TrainCVTestDataSet(train, trainTarget, cv, cvTarget, test, testTarget);
	}

	public DataSet getTrainingSet() {
		return trainingSet;
	}

	public DataSetTarget getTrainingSetTarget() {
		return trainingSetTarget;
	}

	public DataSet getCrossValidationSet() {
		return crossValidationSet;
	}

	public DataSetTarget getCrossValidationSetTarget() {
		return crossValidationSetTarget;
	}

	public DataSet getTestSet() {
		return testSet;
	}

	public DataSetTarget getTestSetTarget() {
		return testSetTarget;
	}

	public FullDataSet getFullTrainingSet() {
		return fullTrainingSet;
	}

	public FullDataSet getFullCrossValidationSet() {
		return fullCrossValidationSet;
	}

	public FullDataSet getFullTestSet() {
		return fullTestSet;
	}
}
