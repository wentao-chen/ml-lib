package com.github.chen.wentao.mllib.data;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.io.PrintStream;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class DataGrapher {

	public static Set<DoublePoint> toDoublePoints(Map<Integer, Double> data) {
		return data.entrySet().stream().map(entry ->
				new DoublePoint(entry.getKey(), entry.getValue())
		).collect(Collectors.toSet());
	}

	public static void graphGui(Map<String, Set<DoublePoint>> data, int width, int height) {
		XYSeriesCollection xyDataSet = new XYSeriesCollection();
		for (Map.Entry<String, Set<DoublePoint>> dataSeries : data.entrySet()) {
			XYSeries series = new XYSeries(dataSeries.getKey());
			dataSeries.getValue().forEach(point -> series.add(point.getX(), point.getY()));
			xyDataSet.addSeries(series);
		}
		JFreeChart chart = ChartFactory.createXYLineChart("Chart title", "X-axis", "Y-axis", xyDataSet, PlotOrientation.VERTICAL, true, true, true);


		JFrame frame = new JFrame("Graph");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.add(new JLabel(new ImageIcon(chart.createBufferedImage(width, height))));
		frame.setSize(width, height);
		frame.setVisible(true);
	}

	public static void graphCli(Set<DoublePoint> data, int width, int height, PrintStream printStream) {
		double xMin = data.stream().mapToDouble(DoublePoint::getX).min().orElse(Double.NaN);
		double xRange = Math.max(data.stream().mapToDouble(DoublePoint::getX).max().orElse(Double.NaN) - xMin, Double.MIN_VALUE);
		double yMin = data.stream().mapToDouble(DoublePoint::getY).min().orElse(Double.NaN);
		double yRange = Math.max(data.stream().mapToDouble(DoublePoint::getY).max().orElse(Double.NaN) - yMin, Double.MIN_VALUE);

		Set<IntPoint> roundedData = data.stream().map(point -> new IntPoint(
				Math.min((int) ((point.getX() - xMin) / xRange * width), width - 1),
				Math.min((int) ((point.getY() - yMin) / yRange * height), height - 1)
		)).collect(Collectors.toSet());

		Set<Integer> existentPoints = new HashSet<>();
		for (IntPoint point : roundedData) {
			existentPoints.add(point.getY() * width + point.getX());
		}

		for (int y = height - 1; y >= 0; y--) {
			double yAxisValue = y * yRange / height + yMin;
			StringBuilder line = new StringBuilder(String.format("%9.3f", yAxisValue*10)).append("|");
			for (int x = 0; x < width; x++) {
				line.append(existentPoints.contains(y * width + x) ? "x" : " ");
			}
			printStream.println(line);
		}
		printStream.println(repeat(" ", 10) + repeat("-", width + 1));
	}

	private static String repeat(String s, int width) {
		StringBuilder str = new StringBuilder();
		for (int i = 0; i < width; i++) {
			str.append(s);
		}
		return str.toString();
	}

	public static class DoublePoint {
		private final double x;
		private final double y;

		private DoublePoint(double x, double y) {
			this.x = x;
			this.y = y;
		}

		public double getX() {
			return x;
		}

		public double getY() {
			return y;
		}

		@Override
		public boolean equals(Object o) {
			if (this == o) return true;
			if (o == null || getClass() != o.getClass()) return false;

			DoublePoint that = (DoublePoint) o;

			if (Double.compare(that.x, x) != 0) return false;
			return Double.compare(that.y, y) == 0;
		}

		@Override
		public int hashCode() {
			int result;
			long temp;
			temp = Double.doubleToLongBits(x);
			result = (int) (temp ^ (temp >>> 32));
			temp = Double.doubleToLongBits(y);
			result = 31 * result + (int) (temp ^ (temp >>> 32));
			return result;
		}
	}

	public static class IntPoint {
		private final int x;
		private final int y;

		private IntPoint(int x, int y) {
			this.x = x;
			this.y = y;
		}

		public int getX() {
			return x;
		}

		public int getY() {
			return y;
		}

		@Override
		public boolean equals(Object o) {
			if (this == o) return true;
			if (o == null || getClass() != o.getClass()) return false;

			IntPoint intPoint = (IntPoint) o;

			if (x != intPoint.x) return false;
			return y == intPoint.y;
		}

		@Override
		public int hashCode() {
			int result = x;
			result = 31 * result + y;
			return result;
		}
	}
}
