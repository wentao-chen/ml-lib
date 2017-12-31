package com.github.chen.wentao.mllib.training;

import org.ejml.simple.SimpleMatrix;

import java.util.Random;

public class TrainCVTestDataSet<T extends DataSetTarget> {

	private final DataSet trainingSet;
	private final T trainingSetTarget;
	private final DataSet crossValidationSet;
	private final T crossValidationSetTarget;
	private final DataSet testSet;
	private final T testSetTarget;

	public TrainCVTestDataSet(DataSet trainingSet, T trainingSetTarget, DataSet crossValidationSet, T crossValidationSetTarget, DataSet testSet, T testSetTarget) {
		this.trainingSet = trainingSet;
		this.trainingSetTarget = trainingSetTarget;
		this.crossValidationSet = crossValidationSet;
		this.crossValidationSetTarget = crossValidationSetTarget;
		this.testSet = testSet;
		this.testSetTarget = testSetTarget;
	}

	public static TrainCVTestDataSet<DataSetTarget> fromFullDataSet(FullDataSet fullDataSet,  Random random) {
		return fromFullDataSet(fullDataSet, random, 0.6, 0.2, 0.2);
	}

	public static TrainCVTestDataSet<DataSetTarget> fromFullDataSet(FullDataSet fullDataSet, Random random, double trainProportion, double cvProportion, double testProportion) {
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
		DataSetTarget trainTarget = new DataSetTarget(targetMatrix.extractMatrix(0, trainSize, 0, targetMatrix.numCols()));
		DataSet cv = new DataSet(dataSetMatrix.extractMatrix(trainSize, trainSize + cvSize, 0, dataSetMatrix.numCols()));
		DataSetTarget cvTarget = new DataSetTarget(targetMatrix.extractMatrix(trainSize, trainSize + cvSize, 0, targetMatrix.numCols()));
		DataSet test = new DataSet(dataSetMatrix.extractMatrix(trainSize + cvSize, dataSetMatrix.numRows(), 0, dataSetMatrix.numCols()));
		DataSetTarget testTarget = new DataSetTarget(targetMatrix.extractMatrix(trainSize + cvSize, targetMatrix.numRows(), 0, targetMatrix.numCols()));
		return new TrainCVTestDataSet<>(train, trainTarget, cv, cvTarget, test, testTarget);
	}

	public DataSet getTrainingSet() {
		return trainingSet;
	}

	public T getTrainingSetTarget() {
		return trainingSetTarget;
	}

	public DataSet getCrossValidationSet() {
		return crossValidationSet;
	}

	public T getCrossValidationSetTarget() {
		return crossValidationSetTarget;
	}

	public DataSet getTestSet() {
		return testSet;
	}

	public T getTestSetTarget() {
		return testSetTarget;
	}
}
