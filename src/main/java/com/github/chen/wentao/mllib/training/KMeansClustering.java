package com.github.chen.wentao.mllib.training;

import com.github.chen.wentao.mllib.util.ejml.SimpleMatrixUtil;
import org.ejml.data.MatrixType;
import org.ejml.simple.SimpleMatrix;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class KMeansClustering {

	public static Map<Integer, Double> kMeansCostPerCluster(DataSet dataSet, Random random, int maxIterations, int[] testCentroidCounts) {
		Map<Integer, Double> costs = new HashMap<>();
		for (int numberOfCentroids : testCentroidCounts) {
			KMeansResult result = kmeans(dataSet, random, numberOfCentroids, maxIterations);
			costs.put(numberOfCentroids, cost(dataSet, result.getClosestCentroids(), result.getCentroids()));
		}
		return costs;
	}

	public static double cost(DataSet dataSet, int[] closestCentroids, SimpleMatrix centroids) {
		assert dataSet.numExamples() == closestCentroids.length;

		SimpleMatrix data = dataSet.getMatrix();

		double sum = 0.0;
		for (int i = 0, m = dataSet.numExamples(); i < m; i++) {
			int c = closestCentroids[i];
			sum += data.rows(i, i + 1).minus(centroids.rows(c, c + 1)).elementPower(2.0).elementSum();
		}
		return sum / dataSet.numExamples();
	}

	public static KMeansResult kmeans(DataSet dataSet, Random random, int numberOfCentroids, int maxIterations) {
		return kmeans(dataSet.getMatrix(), random, numberOfCentroids, maxIterations);
	}

	private static KMeansResult kmeans(SimpleMatrix dataSet, Random random, int numberOfCentroids, int maxIterations) {
		SimpleMatrix centroids = randomlyInitializeCentroids(dataSet, random, numberOfCentroids);
		for (int i = 0; i < maxIterations; i++) {
			int[] closestCentroids = findClosestCentroids(dataSet, centroids);
			centroids = computeCentroids(dataSet, closestCentroids, numberOfCentroids);
		}
		return new KMeansResult(findClosestCentroids(dataSet, centroids), centroids);
	}

	private static SimpleMatrix randomlyInitializeCentroids(SimpleMatrix dataSet, Random random, int numberOfCentroids) {
		assert numberOfCentroids > 0;
		SimpleMatrixUtil.shuffleRows(dataSet, random);
		return dataSet.rows(0, numberOfCentroids);
	}

	private static int[] findClosestCentroids(SimpleMatrix dataSet, SimpleMatrix centroids) {
		assert dataSet.numCols() == centroids.numCols(); // Correct number of features
		assert centroids.numRows() > 0; // At least 1 centroid

		int numCentroids = centroids.numRows();
		int[] closestCentroids = new int[dataSet.numRows()];

		for (int i = 0; i < closestCentroids.length; i++) {
			int closest = 0;
			double closestDistance = dataSet.rows(i, i + 1).minus(centroids.rows(0, 1)).elementPower(2.0).elementSum();
			for (int c = 1; c < numCentroids; c++) {
				double distance = dataSet.rows(i, i + 1).minus(centroids.rows(c, c + 1)).elementPower(2.0).elementSum();
				if (distance < closestDistance) {
					closest = c;
					closestDistance = distance;
				}
			}
			closestCentroids[i] = closest;
		}
		return closestCentroids;
	}

	private static SimpleMatrix computeCentroids(SimpleMatrix dataSet, int[] closestCentroids, int numberOfCentroids) {
		assert dataSet.numRows() == closestCentroids.length; // Correct number of examples

		int n = dataSet.numCols();
		SimpleMatrix centroids = new SimpleMatrix(numberOfCentroids, dataSet.numCols(), MatrixType.DDRM);
		int[] centroidsCount = new int[numberOfCentroids];

		for (int i = 0; i < closestCentroids.length; i++) {
			int c = closestCentroids[i];
			centroidsCount[c] += 1;
			for (int x = 0; x < n; x++) {
				centroids.set(c, x, centroids.get(c, x) + dataSet.get(i, x));
			}
		}
		int centroidIndex = 0;
		for (int c = 0; c < numberOfCentroids; c++) {
			if (centroidsCount[c] == 0) {
				continue;
			}
			for (int x = 0; x < n; x++) {
				centroids.set(centroidIndex, x, centroids.get(c, x) / centroidsCount[c]);
			}
			centroidIndex += 1;
		}
		if (centroidIndex == numberOfCentroids) {
			return centroids;
		}
		return centroids.rows(0, centroidIndex);
	}

	public static class KMeansResult {

		private final int[] closestCentroids;
		private final SimpleMatrix centroids;

		private KMeansResult(int[] closestCentroids, SimpleMatrix centroids) {
			this.closestCentroids = closestCentroids;
			this.centroids = centroids;
		}

		public int[] getClosestCentroids() {
			return Arrays.copyOf(closestCentroids, closestCentroids.length);
		}

		public SimpleMatrix getCentroids() {
			return centroids;
		}
	}
}
