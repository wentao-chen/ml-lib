package com.github.chen.wentao.mllib.training;

@FunctionalInterface
public interface BatchDataSetStream {

    DataSet getBatch(int batchIndex);
}
