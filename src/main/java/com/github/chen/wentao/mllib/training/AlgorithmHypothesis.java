package com.github.chen.wentao.mllib.training;

@FunctionalInterface
public interface AlgorithmHypothesis {
	DataSetTarget apply(DataSet dataSet);
}
