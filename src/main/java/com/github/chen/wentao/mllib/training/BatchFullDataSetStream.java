package com.github.chen.wentao.mllib.training;

public interface BatchFullDataSetStream {

    FullDataSet getBatch(int batchIndex);

    int numBatches();
}
