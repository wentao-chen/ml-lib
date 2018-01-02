package com.github.chen.wentao.mllib.training;

import com.github.chen.wentao.mllib.data.DataUtil;
import com.github.chen.wentao.mllib.util.ejml.SimpleMatrixUtil;
import org.ejml.data.MatrixType;
import org.ejml.simple.SimpleMatrix;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;
import java.util.function.IntFunction;
import java.util.function.Supplier;
import java.util.function.ToIntBiFunction;

import static com.github.chen.wentao.mllib.data.DataUtil.sigmoid;
import static com.github.chen.wentao.mllib.data.DataUtil.sigmoidGrad;
import static com.github.chen.wentao.mllib.util.ejml.SimpleMatrixUtil.ones;

public class NeuralNetwork implements Serializable {

	private static final long serialVersionUID = 4117915678580746872L;

	public static AlgorithmHypothesis getPredictorHypothesis(NeuralNetwork network) {
		return network::predict;
	}

	public static AlgorithmHypothesis getPredictorHypothesis(NeuralNetwork network, double threshold) {
		return dataSet -> network.predict(dataSet, threshold);
	}

	public static AlgorithmHypothesis getMultiPredictorHypothesis(NeuralNetwork network) {
		return network::predictMulti;
	}

	public static SupervisedLearningAlgorithm<NeuralNetwork> getMiniBatchAlgorithm(Supplier<NeuralNetwork> networkGenerator, double alpha, double lambda, int numIterations, ToIntBiFunction<DataSet, DataSetTarget> batchSizeGenerator) {
		return (dataSet, target) -> {
			NeuralNetwork network = networkGenerator.get();
			network.trainMiniBatch(dataSet, target, alpha, lambda, numIterations, batchSizeGenerator.applyAsInt(dataSet, target));
			return network;
		};
	}

	public static SupervisedLearningAlgorithm<NeuralNetwork> getMiniBatchAlgorithm(Supplier<NeuralNetwork> networkGenerator, double alpha, double lambda, int numIterations, int batchSize) {
		return (dataSet, target) -> {
			NeuralNetwork network = networkGenerator.get();
			network.trainMiniBatch(dataSet, target, alpha, lambda, numIterations, batchSize);
			return network;
		};
	}

	public static SupervisedLearningAlgorithm<NeuralNetwork> getAlgorithm(Supplier<NeuralNetwork> networkGenerator, double alpha, double lambda, int numIterations) {
		return (dataSet, target) -> {
			NeuralNetwork network = networkGenerator.get();
			network.train(dataSet, target, alpha, lambda, numIterations);
			return network;
		};
	}

	public static CostFunction<NeuralNetwork> getCostFunction(double lambda) {
		return (network, dataSet, target) -> network.costFunction(dataSet, target, lambda);
	}

	private final SimpleMatrix[] thetas;

	public NeuralNetwork(SimpleMatrix[] thetas) {
		for (int i = 1; i < thetas.length; i++) {
			assert(thetas[i - 1].numRows() + 1 == thetas[i].numCols()); // correct number of neurons per layer (with bias)
		}

		this.thetas = thetas;
	}

	public static NeuralNetwork emptyNetwork(int... numNeurons) {
		assert numNeurons.length >= 2;

		SimpleMatrix[] thetas = new SimpleMatrix[numNeurons.length - 1];
		for (int i = 0; i < thetas.length; i++) {
			thetas[i] = new SimpleMatrix(numNeurons[i + 1], numNeurons[i] + 1, MatrixType.DDRM);
		}
		return new NeuralNetwork(thetas);
	}

	/**
	 * Randomly initializes all neuron weights of {@code this} neural network
	 * @param random the random instance used
	 * @return {@code this}
	 */
	public NeuralNetwork randomlyInitialize(Random random) {
		for (SimpleMatrix theta : thetas) {
			double epsilon = Math.sqrt(6.0 / (theta.numCols() + theta.numRows()));
			double epsilonDouble = 2.0 * epsilon;
			for (int i = theta.getNumElements() - 1; i >= 0; i--) {
				theta.set(i, random.nextDouble() * epsilonDouble - epsilon);
			}
		}
		return this;
	}

	/**
	 * Randomly initializes all neuron weights of {@code this} neural network
	 * @param initEpsilon the maximum absolute value of each neuron weight
	 * @param random the random instance used
	 * @return {@code this}
	 */
	public NeuralNetwork randomlyInitialize(double initEpsilon, Random random) {
		assert initEpsilon > 0;

		double epsilonDouble = 2.0 * initEpsilon;
		for (SimpleMatrix theta : thetas) {
			for (int i = theta.getNumElements() - 1; i >= 0; i--) {
				theta.set(i, random.nextDouble() * epsilonDouble - initEpsilon);
			}
		}
		return this;
	}

	public int numLayers() {
		return thetas.length + 1;
	}

	public int numInputs() {
		return numNeurons(0);
	}

	public int numOutputs() {
		return numNeurons(numLayers() - 1);
	}

	public int numNeurons(int layer) {
		return layer == 0 ? thetas[0].numCols() - 1 : thetas[layer - 1].numRows();
	}

	public double costFunction(DataSet dataSet, DataSetTarget target, double lambda) {
		return costFunction(thetas, dataSet.getMatrix(), targetToMatrix(target), lambda);
	}

	private static double costFunction(SimpleMatrix[] thetas, SimpleMatrix dataSet, SimpleMatrix target, double lambda) {
		assert(dataSet.numCols() + 1 == thetas[0].numCols()); // correct number of input features
		for (int i = 1; i < thetas.length; i++) {
			assert(thetas[i - 1].numRows() + 1 == thetas[i].numCols()); // correct number of neurons per layer (with bias)
		}
		assert(thetas[thetas.length - 1].numRows() == target.numCols()); // correct number of output neurons
		assert(dataSet.numRows() == target.numRows()); // correct number of training examples
		assert(lambda >= 0 && Double.isFinite(lambda));

		double m = dataSet.numRows();

		SimpleMatrix[] activations = NeuralNetwork.feedForward(thetas, dataSet);
		SimpleMatrix lastLayer = activations[activations.length - 1].transpose(); // For convenience

		// Compute cost
		double regularizationCost = 0.0;
		if (lambda > 0) {
			for (SimpleMatrix theta : thetas) {
				regularizationCost += theta.extractMatrix(0, theta.numRows(), 1, theta.numCols()).elementPower(2.0).elementSum();
			}
			regularizationCost *= lambda / 2.0;
		}
		return (target.negative().elementMult(lastLayer.elementLog()).minus(ones(target).minus(target).elementMult(ones(lastLayer).minus(lastLayer).elementLog())).elementSum() + regularizationCost) / m;
	}

	public DataSetTarget predict(DataSet dataSet) {
		return numOutputs() == 1 ? predict(dataSet, 0.5) : predictMulti(dataSet);
	}

	public DataSetTarget predict(DataSet dataSet, double threshold) {
		return new DataSetTarget(predict(thetas, dataSet.getMatrix(), threshold), 2);
	}

	public DataSetTarget predictMulti(DataSet dataSet) {
		return new DataSetTarget(predictMulti(thetas, dataSet.getMatrix()), numOutputs());
	}

	private static SimpleMatrix predict(SimpleMatrix[] thetas, SimpleMatrix dataSet, double threshold) {
		double[] predictions = new double[dataSet.numRows()];
		SimpleMatrix output = compute(thetas, dataSet);
		for (int i = 0, m = dataSet.numRows(); i < m; i++) {
			predictions[i] = output.get(i, 0) >= threshold ? 1.0 : 0.0;
		}
		return new SimpleMatrix(predictions.length, 1, true, predictions);
	}

	private static SimpleMatrix predictMulti(SimpleMatrix[] thetas, SimpleMatrix dataSet) {
		double[] predictions = new double[dataSet.numRows()];
		SimpleMatrix output = compute(thetas, dataSet);
		int outputs = output.numCols();
		for (int i = 0, m = dataSet.numRows(); i < m; i++) {
			int max = 0;
			double maxValue = output.get(i, 0);
			for (int j = 1; j < outputs; j++) {
				double value = output.get(i, j);
				if (value > maxValue) {
					max = j;
					maxValue = value;
				}
			}
			predictions[i] = max;
		}
		return new SimpleMatrix(predictions.length, 1, true, predictions);
	}

	public SimpleMatrix compute(DataSet dataSet) {
		return compute(thetas, dataSet.getMatrix());
	}

	private static SimpleMatrix compute(SimpleMatrix[] thetas, SimpleMatrix dataSet) {
		SimpleMatrix[] activations = feedForward(thetas, dataSet);
		return activations[activations.length - 1].transpose();
	}

	/**
	 * Computes the activation values for each neuron given a data set
	 * @param dataSet (m) x (s0) matrix of training examples where s0 is the number of input neurons
	 * @return a ({@link #numLayers()})-length array of (si) x (m) matrices where si is the number of neurons in layer i of each neuron activation value
	 */
	public SimpleMatrix[] feedForward(DataSet dataSet) {
		return feedForward(thetas, dataSet.getMatrix());
	}

	private static SimpleMatrix[] feedForward(SimpleMatrix[] thetas, SimpleMatrix dataSet) {
		assert(dataSet.numCols() + 1 == thetas[0].numCols()); // correct number of input features
		for (int i = 1; i < thetas.length; i++) {
			assert(thetas[i - 1].numRows() + 1 == thetas[i].numCols()); // correct number of neurons per layer (with bias)
		}

		final int layers = thetas.length + 1;

		// Feed forward
		SimpleMatrix[] activations = new SimpleMatrix[layers];
		activations[0] = DataUtil.addBiasColumn(dataSet).transpose();
		for (int i = 1; i < layers - 1; i++) {
			activations[i] = DataUtil.addBiasRow(sigmoid(thetas[i - 1].mult(activations[i - 1])));
		}
		activations[layers - 1] = sigmoid(thetas[layers - 2].mult(activations[layers - 2]));
		return activations;
	}

	/**
	 * Computes the gradients for each neuron given a data set and the target values
	 * @param dataSet the data set used to compute the gradients
	 * @param target the target values for the data set
	 * @param lambda the regularization parameter (greater or equal to 0)
	 * @return a ({@link #numLayers()})-length array of (m) x (si) matrices where si is the number of neurons in layer i of each neuron gradient value
	 */
	public SimpleMatrix[] backPropagation(DataSet dataSet, DataSetTarget target, double lambda) {
		return backPropagation(thetas, dataSet.getMatrix(), targetToMatrix(target), lambda);
	}

	private static SimpleMatrix[] backPropagation(SimpleMatrix[] thetas, SimpleMatrix dataSet, SimpleMatrix target, double lambda) {
		assert(dataSet.numCols() + 1 == thetas[0].numCols()); // correct number of input features
		for (int i = 1; i < thetas.length; i++) {
			assert(thetas[i - 1].numRows() + 1 == thetas[i].numCols()); // correct number of neurons per layer (with bias)
		}
		assert(thetas[thetas.length - 1].numRows() == target.numCols()); // correct number of output neurons
		assert(dataSet.numRows() == target.numRows()); // correct number of training examples
		assert(lambda >= 0 && Double.isFinite(lambda));

		SimpleMatrix[] activations = feedForward(thetas, dataSet);

		// Init
		SimpleMatrix[] deltas = new SimpleMatrix[thetas.length];
		SimpleMatrix[] grads = new SimpleMatrix[thetas.length];
		for (int i = 0; i < deltas.length; i++) {
			int rows = thetas[i].numRows();
			int cols = thetas[i].numCols();
			deltas[i] = new SimpleMatrix(rows, cols, MatrixType.DDRM);
			grads[i] = new SimpleMatrix(rows, cols, MatrixType.DDRM);
		}

		// Calculate gradients
		for (int i = 0, m = dataSet.numRows(); i < m; i++) {
			deltas[deltas.length - 1] = activations[deltas.length].cols(i, i + 1).minus(target.rows(i, i + 1).transpose());
			for (int j = deltas.length - 2; j >= 0; j--) {
				deltas[j] = thetas[j + 1].transpose().mult(deltas[j + 1]).elementMult(sigmoidGrad(activations[j + 1].cols(i, i + 1))).rows(1, activations[j + 1].numRows());
			}
			for (int j = 0; j < grads.length; j++) {
				grads[j] = grads[j].plus(deltas[j].mult(activations[j].cols(i, i + 1).transpose()));
			}
		}

		double m = dataSet.numRows();
		for (int i = 0; i < grads.length; i++) {
			if (lambda > 0) {
				grads[i] = grads[i].plus(SimpleMatrixUtil.setColumn(thetas[i], 0, 0.0).scale(lambda));
			}
			grads[i] = grads[i].divide(m);
		}
		return grads;
	}

	public SimpleMatrix[] numericalGradient(DataSet dataSet, DataSetTarget target, double lambda) {
		return numericalGradient(dataSet, target, lambda, 1.0e-4);
	}

	public SimpleMatrix[] numericalGradient(DataSet dataSet, DataSetTarget target, double lambda, double epsilon) {
		return numericalGradient(thetas, dataSet.getMatrix(), targetToMatrix(target), lambda, epsilon);
	}

	private static SimpleMatrix[] numericalGradient(SimpleMatrix[] thetasOriginal, SimpleMatrix dataSet, SimpleMatrix target, double lambda, double epsilon) {
		assert(dataSet.numCols() + 1 == thetasOriginal[0].numCols()); // correct number of input features
		for (int i = 1; i < thetasOriginal.length; i++) {
			assert(thetasOriginal[i - 1].numRows() + 1 == thetasOriginal[i].numCols()); // correct number of neurons per layer (with bias)
		}
		assert(thetasOriginal[thetasOriginal.length - 1].numRows() == target.numCols()); // correct number of output neurons
		assert(dataSet.numRows() == target.numRows()); // correct number of training examples
		assert(lambda >= 0 && Double.isFinite(lambda));
		assert(epsilon > 0 && Double.isFinite(epsilon));

		double epsilonDouble = epsilon * 2.0;

		SimpleMatrix[] thetas = Arrays.copyOf(thetasOriginal, thetasOriginal.length);

		SimpleMatrix[] grads = new SimpleMatrix[thetas.length];
		for (int layer = 0; layer < grads.length; layer++) {
			SimpleMatrix theta = thetas[layer] = thetas[layer].copy();
			SimpleMatrix grad = new SimpleMatrix(theta.numRows(), theta.numCols(), MatrixType.DDRM);
			for (int i = 0, n = theta.getNumElements(); i < n; i++) {
				double original = theta.get(i);
				theta.set(i, original - epsilon);
				double cost1 = costFunction(thetas, dataSet, target, lambda);
				theta.set(i, original + epsilon);
				double cost2 = costFunction(thetas, dataSet, target, lambda);
				grad.set(i, (cost2 - cost1) / epsilonDouble);
			}
			grads[layer] = grad;
		}
		return grads;
	}

	public void train(DataSet dataSet, DataSetTarget target, double alpha, double lambda, int numIterations) {
		train(this.thetas, dataSet.getMatrix(), targetToMatrix(target), alpha, lambda, numIterations);
	}

	public void print(int numChar, int precision) {
		for (SimpleMatrix theta : thetas) {
			theta.print(numChar, precision);
		}
	}

	private static void train(SimpleMatrix[] thetas, SimpleMatrix dataSet, SimpleMatrix target, double alpha, double lambda, int numIterations) {
		assert (alpha > 0 && Double.isFinite(alpha));

		for (int i = 0; i < numIterations; i++) {
			SimpleMatrix[] grad = backPropagation(thetas, dataSet, target, lambda);
			for (int layer = 0; layer < thetas.length; layer++) {
				thetas[layer] = thetas[layer].minus(grad[layer].scale(alpha));
			}
		}
	}

	public void trainStochastic(DataSet dataSet, DataSetTarget target, double alpha, double lambda, int numIterations) {
		trainMiniBatch(dataSet, target, alpha, lambda, numIterations, 1);
	}

	public void trainMiniBatch(DataSet dataSet, DataSetTarget target, double alpha, double lambda, int numIterations, int batchSize) {
		if (batchSize == dataSet.numExamples()) {
			train(this.thetas, dataSet.getMatrix(), targetToMatrix(target), alpha, lambda, numIterations);
		} else {
			trainMiniBatch(this.thetas, dataSet.getMatrix(), targetToMatrix(target), alpha, lambda, numIterations, batchSize);
		}
	}

	private static void trainMiniBatch(SimpleMatrix[] thetas, SimpleMatrix dataSet, SimpleMatrix target, double alpha, double lambda, int numIterations, int batchSize) {
		assert (alpha > 0 && Double.isFinite(alpha));

		int m = dataSet.numRows();
		for (int i = 0, batchIndex = 0; i < numIterations; i++, batchIndex++) {
			int rowsStart = batchIndex * batchSize;
			int rowsEnd = rowsStart + batchSize;
			if (rowsEnd >= m) {
				rowsEnd = m;
				batchIndex = 0;
			}
			SimpleMatrix dataSetBatch = dataSet.rows(rowsStart, rowsEnd);
			SimpleMatrix targetBatch = target.rows(rowsStart, rowsEnd);
			SimpleMatrix[] grad = backPropagation(thetas, dataSetBatch, targetBatch, lambda);
			for (int layer = 0; layer < thetas.length; layer++) {
				thetas[layer] = thetas[layer].minus(grad[layer].scale(alpha));
			}
		}
	}

    public void trainMiniBatch(IntFunction<FullDataSet> batchGenerator, double alpha, double lambda, int numIterations) {
        trainMiniBatch(this.thetas, batchGenerator, alpha, lambda, numIterations);
    }

    private void trainMiniBatch(SimpleMatrix[] thetas, IntFunction<FullDataSet> batchGenerator, double alpha, double lambda, int numIterations) {
        assert (alpha > 0 && Double.isFinite(alpha));

        for (int i = 0; i < numIterations; i++) {
            FullDataSet batch = batchGenerator.apply(i);
            SimpleMatrix dataSetBatch = batch.getDataSet().getMatrix();
            SimpleMatrix targetBatch = targetToMatrix(batch.getDataSetTarget());
            SimpleMatrix[] grad = backPropagation(thetas, dataSetBatch, targetBatch, lambda);
            for (int layer = 0; layer < thetas.length; layer++) {
                thetas[layer] = thetas[layer].minus(grad[layer].scale(alpha));
            }
        }
    }

	public static NeuralNetwork loadFromFileBinary(String directoryName) throws IOException {
		String[] fileNames = new File(directoryName).list((dir, name) -> name.toLowerCase().endsWith(".nnbin"));
		if (fileNames == null) throw new IOException();
		SimpleMatrix[] thetas = new SimpleMatrix[fileNames.length];
		for (int i = 0; i < thetas.length; i++) {
			thetas[i] = SimpleMatrix.loadBinary(directoryName + "/" + i + ".nnbin");
		}
		return new NeuralNetwork(thetas);
	}

	public void saveToFileBinary(String directory) throws IOException {
		for (int i = 0; i < thetas.length; i++) {
			thetas[i].saveToFileBinary(directory + "/" + i + ".nnbin");
		}
	}

	private SimpleMatrix targetToMatrix(DataSetTarget target) {
		return numOutputs() == 1 ? target.getMatrix() : target.toBinaryMatrix();
	}
}
