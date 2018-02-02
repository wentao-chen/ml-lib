package com.github.chen.wentao.mllib.training;

public abstract class StaticFullDataSetStream implements BatchFullDataSetStream {

    private final int numBatches;

    protected StaticFullDataSetStream(int numBatches) {
        this.numBatches = numBatches;
    }

    @Override
    public int numBatches() {
        return numBatches;
    }
}
