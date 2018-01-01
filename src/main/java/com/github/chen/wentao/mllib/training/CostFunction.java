package com.github.chen.wentao.mllib.training;

@FunctionalInterface
public interface CostFunction<T> {
	double apply(T modelParameters, DataSet dataSet, DataSetTarget dataSetTarget);
}
