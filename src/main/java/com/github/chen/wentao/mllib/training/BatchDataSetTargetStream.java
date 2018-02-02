package com.github.chen.wentao.mllib.training;

@FunctionalInterface
public interface BatchDataSetTargetStream {

    DataSetTarget getBatch(int batchIndex);
}
