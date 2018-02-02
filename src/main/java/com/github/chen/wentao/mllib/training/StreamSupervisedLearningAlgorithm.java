package com.github.chen.wentao.mllib.training;

@FunctionalInterface
public interface StreamSupervisedLearningAlgorithm<T> {

	T findOptimalParameters(BatchFullDataSetStream dataSet);
}
