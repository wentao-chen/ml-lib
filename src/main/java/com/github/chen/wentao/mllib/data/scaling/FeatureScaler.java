package com.github.chen.wentao.mllib.data.scaling;

import com.github.chen.wentao.mllib.training.DataSet;

public interface FeatureScaler {

	DataSet getNormalizedDataSet();

	DataSet normalize(DataSet data);
}
