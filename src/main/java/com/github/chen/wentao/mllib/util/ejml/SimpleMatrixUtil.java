package com.github.chen.wentao.mllib.util.ejml;

import org.ejml.data.MatrixType;
import org.ejml.simple.SimpleMatrix;

import java.util.Random;
import java.util.stream.IntStream;

public class SimpleMatrixUtil {

	public static SimpleMatrix shortDiagonal(int width, double value) {
		double[] values = new double[width];
		for (int i = 1; i < width; i++) {
			values[i] = value;
		}
		return SimpleMatrix.diag(values);
	}

	public static SimpleMatrix ones(SimpleMatrix m) {
		return ones(m.numRows(), m.numCols());
	}

	public static SimpleMatrix ones(int numRows, int numCols) {
		SimpleMatrix matrix = new SimpleMatrix(numRows, numCols, MatrixType.DDRM);
		matrix.set(1.0);
		return matrix;
	}

	public static SimpleMatrix sumCols(SimpleMatrix matrix) {
		return ones(1, matrix.numRows()).mult(matrix);
	}

	public static SimpleMatrix colMin(SimpleMatrix data) {
		assert(data.numRows() > 0);

		double[] mins = new double[data.numCols()];
		for (int col = data.numCols() - 1; col >= 0; col--) {
			double min = Double.POSITIVE_INFINITY;
			for (int row = data.numRows() - 1; row >= 0; row--) {
				min = Math.min(min, data.get(row, col));
			}
			mins[col] = min;
		}
		return new SimpleMatrix(1, mins.length, true, mins);
	}

	public static SimpleMatrix colMax(SimpleMatrix data) {
		assert(data.numRows() > 0);

		double[] maxs = new double[data.numCols()];
		for (int col = data.numCols() - 1; col >= 0; col--) {
			double max = Double.POSITIVE_INFINITY;
			for (int row = data.numRows() - 1; row >= 0; row--) {
				max = Math.max(max, data.get(row, col));
			}
			maxs[col] = max;
		}
		return new SimpleMatrix(1, maxs.length, true, maxs);
	}

	public static SimpleMatrix filterEquals(SimpleMatrix target, double value) {
		double[] data = new double[target.getNumElements()];
		for (int i = 0; i < data.length; i++) {
			data[i] = target.get(i) == value ? 1.0 : 0.0;
		}
		return new SimpleMatrix(target.numRows(), target.numCols(), true, data);
	}

	public static SimpleMatrix setColumn(SimpleMatrix m, int column, double value) {
		double[] values = new double[m.numRows()];
		for (int i = 0; i < values.length; i++) {
			values[i] = value;
		}
		SimpleMatrix result = m.copy();
		result.setColumn(column, 0, values);
		return result;
	}

	public static SimpleMatrix shuffleRows(SimpleMatrix m, Random random) {
		int cols = m.numCols();
		int rows = m.numRows();
		SimpleMatrix newMatrix = new SimpleMatrix(rows, cols, m.getType());
		int[] copyRows = newShuffledArray(rows, random);
		for (int y = 0; y < rows; y++) {
			for (int x = 0; x < cols; x++) {
				newMatrix.set(y, x, m.get(copyRows[y], x));
			}
		}
		return newMatrix;
	}

	private static int[] newShuffledArray(int size, Random random) {
		int[] array = IntStream.range(0, size).toArray();
		for (int i = size - 1; i >= 0; i--) {
			int swapIndex = random.nextInt(i + 1);
			int temp = array[i];
			array[i] = array[swapIndex];
			array[swapIndex] = temp;
		}
		return array;
	}
}
