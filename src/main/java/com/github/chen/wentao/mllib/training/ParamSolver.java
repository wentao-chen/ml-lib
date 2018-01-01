package com.github.chen.wentao.mllib.training;

import java.util.Comparator;
import java.util.function.Function;
import java.util.stream.Stream;

public class ParamSolver {

	public static <T, S> T findOptimalParams(Function<T, SupervisedLearningAlgorithm<S>> learningAlgorithmGenerator, Function<T, CostFunction<S>> costFunctionGenerator, Function<T, FullDataSet> trainingDataSetGenerator, FullDataSet crossValidationDataSet, Stream<T> testValues) {
		DataSet cvDataSet = crossValidationDataSet.getDataSet();
		DataSetTarget cvDataSetTarget = crossValidationDataSet.getDataSetTarget();
		return testValues.min(Comparator.comparingDouble(
				v -> cvCost(learningAlgorithmGenerator, costFunctionGenerator, trainingDataSetGenerator, cvDataSet, cvDataSetTarget, v)
		)).orElse(null);
	}

	private static <T, S> double cvCost(Function<T, SupervisedLearningAlgorithm<S>> learningAlgorithmGenerator, Function<T, CostFunction<S>> costFunctionGenerator, Function<T, FullDataSet> trainingDataSetGenerator, DataSet cvDataSet, DataSetTarget cvDataSetTarget, T value) {
		FullDataSet trainDataSubset = trainingDataSetGenerator.apply(value);
		SupervisedLearningAlgorithm<S> algorithm = learningAlgorithmGenerator.apply(value);
		CostFunction<S> costFunction = costFunctionGenerator.apply(value);
		S optimalParamsTrain = algorithm.findOptimalParameters(trainDataSubset.getDataSet(), trainDataSubset.getDataSetTarget());
		return costFunction.apply(optimalParamsTrain, cvDataSet, cvDataSetTarget);
	}

	public static <T, S> T findOptimalParams(Function<T, SupervisedLearningAlgorithm<S>> learningAlgorithmGenerator, Function<T, CostFunction<S>> costFunctionGenerator, TrainCVTestDataSet fullDataSet, Stream<T> testValues) {
		return testValues.min(Comparator.comparingDouble(
				v -> cvCost(learningAlgorithmGenerator, costFunctionGenerator, fullDataSet, v)
		)).orElse(null);
	}

	private static <T, S> double cvCost(Function<T, SupervisedLearningAlgorithm<S>> learningAlgorithmGenerator, Function<T, CostFunction<S>> costFunctionGenerator, TrainCVTestDataSet fullDataSet, T value) {
		FullDataSet trainDataSubset = fullDataSet.getFullTrainingSet();
		DataSet cvDataSet = fullDataSet.getCrossValidationSet();
		DataSetTarget cvDataSetTarget = fullDataSet.getCrossValidationSetTarget();
		SupervisedLearningAlgorithm<S> algorithm = learningAlgorithmGenerator.apply(value);
		CostFunction<S> costFunction = costFunctionGenerator.apply(value);
		S optimalParamsTrain = algorithm.findOptimalParameters(trainDataSubset.getDataSet(), trainDataSubset.getDataSetTarget());
		return costFunction.apply(optimalParamsTrain, cvDataSet, cvDataSetTarget);
	}
}
