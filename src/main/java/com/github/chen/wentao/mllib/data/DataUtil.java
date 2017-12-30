package com.github.chen.wentao.mllib.data;

import com.github.chen.wentao.mllib.util.ejml.SimpleMatrixUtil;
import org.ejml.simple.SimpleMatrix;

public class DataUtil {

	public static SimpleMatrix addBiasColumn(SimpleMatrix data) {
		return SimpleMatrixUtil.ones(data.numRows(), 1).concatColumns(data);
	}
}
