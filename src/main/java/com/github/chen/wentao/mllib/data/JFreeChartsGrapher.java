package com.github.chen.wentao.mllib.data;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.LogAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.WindowConstants;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.BorderLayout;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class JFreeChartsGrapher {

    private final Map<Double, Double> data = new HashMap<>();

    public void addDatum(double x, double y) {
        data.put(x, y);
    }

    public static JFreeChartsGrapher fromData(List<Double> data) {
        JFreeChartsGrapher jFreeChartsGrapher = new JFreeChartsGrapher();
        for (int i = 0; i < data.size(); i++) {
            jFreeChartsGrapher.addDatum(i, data.get(i));
        }
        return jFreeChartsGrapher;
    }

    public JFreeChart graphToChart(String title, String xAxisLabel, String yAxisLabel, boolean useLogAxis) {
        XYSeries series = new XYSeries("Data");
        data.forEach(series::add);

        XYSeriesCollection xyDataSet = new XYSeriesCollection();
        if (!data.isEmpty()) xyDataSet.addSeries(series);
        JFreeChart chart = ChartFactory.createXYLineChart(title, xAxisLabel, yAxisLabel, xyDataSet, PlotOrientation.VERTICAL, true, true, true);
        if (useLogAxis) chart.getXYPlot().setDomainAxis(new LogAxis(xAxisLabel));
        return chart;
    }

    public void graphWithJFrame(String title, String xAxisLabel, String yAxisLabel, boolean useLogAxis, int width, int height) {
        JFreeChart chart = graphToChart(title, xAxisLabel, yAxisLabel, useLogAxis);
        JFrame frame = new JFrame("Learning Curves");
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        JPanel mainPanel = new JPanel(new BorderLayout());
        mainPanel.add(new ChartPanel(chart), BorderLayout.CENTER);
        JPanel buttonPanel = new JPanel(new BorderLayout());
        JButton saveButton = new JButton("Save");
        saveButton.addActionListener(e -> {
            JFileChooser fileChooser = new JFileChooser(System.getProperty("user.dir"));
            fileChooser.setFileFilter(new FileNameExtensionFilter(".png", "png"));
            if (fileChooser.showSaveDialog(frame) == JFileChooser.APPROVE_OPTION) {
                String fileName = fileChooser.getSelectedFile().toString();
                if (!fileName.toLowerCase().endsWith(".png")) {
                    fileName += ".png";
                }
                File file = new File(fileName);
                if (file.exists()) {
                    if (JOptionPane.showConfirmDialog(frame, "Overwrite existing: " + fileName) != JOptionPane.YES_OPTION) {
                        return;
                    }
                }
                try {
                    ChartUtilities.saveChartAsPNG(file, chart, width, height);
                    JOptionPane.showMessageDialog(frame, "Saved");
                } catch (IOException e1) {
                    JOptionPane.showMessageDialog(frame, "IOException " + e1);
                }
            }
        });
        buttonPanel.add(saveButton, BorderLayout.EAST);
        mainPanel.add(buttonPanel, BorderLayout.SOUTH);
        frame.add(mainPanel);
        frame.setSize(width, height);
        frame.setVisible(true);
    }
}
