package com.github.chen.wentao.mllib.training;

import org.ejml.simple.SimpleMatrix;

public class FeatureParameters implements MatrixWrapper {

	private final SimpleMatrix theta;

	public FeatureParameters(double... data) {
		this(new SimpleMatrix(data.length, 1, true, data));
	}

	public FeatureParameters(SimpleMatrix theta) {
		assert(theta.numCols() == 1); // is vector

		this.theta = theta;
	}

	@Override
	public SimpleMatrix getMatrix() {
		return theta;
	}

	public double get(int featureIndex) {
		return theta.get(featureIndex);
	}

	public int numFeatures() {
		return theta.numRows();
	}
}
