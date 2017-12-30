package com.github.chen.wentao.mllib.training;

import org.ejml.simple.SimpleMatrix;

public class LabeledDataSetTarget extends DataSetTarget {

	private Integer labels = null;

	public LabeledDataSetTarget(int... data) {
		this(new SimpleMatrix(data.length, 1, true, intArrayToDouble(data)));
	}

	public LabeledDataSetTarget(SimpleMatrix theta) {
		super(theta);
		assertValidValues();
	}

	public int getLabel(int exampleIndex) {
		return (int) get(exampleIndex);
	}

	private static double[] intArrayToDouble(int[] data) {
		double[] array = new double[data.length];
		for (int i = 0; i < data.length; i++) {
			array[i] = data[i];
		}
		return array;
	}

	public int numLabels() {
		if (labels == null) {
			labels = countLabels();
		}
		return labels;
	}

	private int countLabels() {
		int max = 0;
		SimpleMatrix matrix = getMatrix();
		for (int i = matrix.getNumElements() - 1; i >= 0; i--) {
			max = Math.max(max, (int) matrix.get(i));
		}
		return max + 1;
	}

	private void assertValidValues() {
		SimpleMatrix matrix = getMatrix();
		for (int i = matrix.getNumElements() - 1; i >= 0; i--) {
			assert(matrix.get(i) % 1.0 == 0.0);
		}
	}
}
