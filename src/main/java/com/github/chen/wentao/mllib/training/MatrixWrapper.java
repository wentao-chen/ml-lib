package com.github.chen.wentao.mllib.training;

import org.ejml.simple.SimpleMatrix;

public interface MatrixWrapper {

	SimpleMatrix getMatrix();

	default void print(int numChars, int precision) {
		getMatrix().print(numChars, precision);
	}
}
