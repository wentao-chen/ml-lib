package com.github.chen.wentao.mllib.training;

@FunctionalInterface
public interface SupervisedLearningAlgorithm<T> {

	T findOptimalParameters(DataSet dataSet, DataSetTarget target);
}
