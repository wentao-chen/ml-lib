package com.github.chen.wentao.mllib.training;

@FunctionalInterface
public interface StreamCostFunction<T> {
	double apply(T modelParameters, BatchFullDataSetStream dataSetStream);
}
