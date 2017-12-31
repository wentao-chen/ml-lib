package com.github.chen.wentao.mllib.training;

import com.github.chen.wentao.mllib.util.ejml.SimpleMatrixUtil;
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

	public SimpleMatrix toBinaryMatrix() {
		SimpleMatrix initMatrix = getMatrix();
		SimpleMatrix targetBinaryMatrix = SimpleMatrixUtil.filterEquals(initMatrix, 0);
		for (int i = 1, labels = numLabels(); i < labels; i++) {
			targetBinaryMatrix = targetBinaryMatrix.concatColumns(SimpleMatrixUtil.filterEquals(initMatrix, i));
		}
		return targetBinaryMatrix;
	}

	public int getLabel(int exampleIndex) {
		return (int) get(exampleIndex);
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

	private static double[] intArrayToDouble(int[] data) {
		double[] array = new double[data.length];
		for (int i = 0; i < data.length; i++) {
			array[i] = data[i];
		}
		return array;
	}
}
