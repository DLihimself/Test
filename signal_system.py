import copy
import configparser
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

import simulator.data_simulator as dtsm
from modules.detect_module import DetectModule
from modules.inference_module import InferenceModule
from modules.repair_module import RepairModule


DEFAULT_CFG_FILE = 'config/setup.ini'


class SignalSystem():
    """
    光信号系统：主要由信号检测模块和信号修复模块组成
    """

    def __init__(self, cfg_file=DEFAULT_CFG_FILE):
        print('system init')

        # 读取系统配置文件
        self.config = self.init_config(cfg_file)

        # 初始化信号检测模块
        self.detect_module = DetectModule()

        # 初始化信号诊断模块
        self.infer_module = InferenceModule()

        # 初始化信号修复模块
        self.repair_module = RepairModule()

        self.load_module()

        self.train_data = None
        self.input_data = None

    def init_config(self, cfg_file):
        config = configparser.ConfigParser()
        config.read(cfg_file)
        return config

    def get_simulate_data(self, data_type):
        n_samples = self.config.getint(data_type, 'n_samples', fallback=1e5)
        fault_ratio = self.config.getfloat(data_type, 'fault_ratio', fallback=0.2)
        save_path = self.config.get(data_type, 'save_path', fallback=None)
        data = dtsm.simulate_data(n_samples=n_samples, fault_ratio=fault_ratio, is_full=True)

        if save_path is not None:
            if data_type == 'test_data':
                save_data = copy.deepcopy(data)
                save_data.reindex(columns=dtsm.TEST_FEATURE_COL)
            else:
                save_data = data
            save_data.to_csv(save_path, index=True)

        return data

    def load_data(self, load_path):
        return pd.read_csv(load_path, index_col=0)

    def get_train_data(self):
        load_path = self.config.get('train_data', 'load_path', fallback=None)
        if load_path is not None:
            data = self.load_data(load_path)
        else:
            data = self.get_simulate_data('train_data')

        data_filtered = data.reindex(columns=dtsm.FEATURE_COL + ['is_fault'])
        '''
        data_filtered = data.reindex(columns=dtsm.LOWACC_FEATURE_COL + ['is_fault'])

        label_encoder = LabelEncoder()
        str_cols = ['fiberType', 'encoding']
        for col in str_cols:
          data_filtered[col] = label_encoder.fit_transform(data_filtered[col])
        '''

        return data_filtered

    def get_test_data(self):
        load_path = self.config.get('test_data', 'load_path', fallback=None)
        if load_path is not None:
            data = self.load_data(load_path)
        else:
            data = self.get_simulate_data('test_data')

        data_filtered = data.reindex(columns=dtsm.FEATURE_COL)

        return data, data_filtered

    def get_detect_train_data(self):
        if self.train_data is None:
            self.train_data = self.get_train_data()

        return self.train_data

    def check_full_data(self, data):
        is_valid, error_features = dtsm.check_full_data(data)
        return is_valid, error_features

    def get_tensor_data(self, data_pd):
        # 将DataFrame的数据转换为tensor类型的数据
        data_array = np.array(data_pd)
        data_tensor = torch.tensor(data_array, dtype=torch.float32)
        return data_tensor.reshape((-1, 1, 5))

    def load_module(self):
        detect_load_path = self.config.get('detect_model', 'load_path', fallback=None)
        if detect_load_path is not None:
            self.detect_module.load_model(detect_load_path)

        repair_load_path = self.config.get('repair_model', 'load_path', fallback=None)
        if repair_load_path is not None:
            self.repair_module.load_model(repair_load_path)


    def train_detect_module(self):
        # 获取训练数据
        train_data = self.get_detect_train_data()

        # 获取训练的配置
        batch_size = self.config.getint('detect_model', 'train_batch_size')
        epochs = self.config.getint('detect_model', 'train_epochs')

        # 训练检测模块
        train_results, test_results = self.detect_module.train(train_data, batch_size=batch_size, epochs=epochs)
        log_path = self.config.get('detect_model', 'log_dir', fallback='logs/')
        train_results.to_csv(log_path+'detect_model_train_results.csv', index=True)
        test_results.to_csv(log_path+'detect_model_test_results.csv', index=True)

        # 保存模型参数
        save_path = self.config.get('detect_model', 'save_path', fallback=None)
        if save_path is not None:
            self.detect_module.save_model(save_path)

    def get_repair_train_data(self):
        if self.train_data is None:
            self.train_data = self.get_train_data()
        data_normal = self.train_data[self.train_data['is_fault']==0]
        train_data = data_normal.reindex(columns=dtsm.FEATURE_COL)
        
        return dtsm.get_normalized_data(train_data)

    def train_repair_module(self):
        # 获取训练数据
        train_data = self.get_repair_train_data()

        # 获取训练的配置
        batch_size = self.config.getint('repair_model', 'train_batch_size')
        epochs = self.config.getint('repair_model', 'train_epochs')

        # 训练修复模块
        train_results = self.repair_module.train(train_data, batch_size=batch_size, epochs=epochs)
        log_path = self.config.get('repair_model', 'log_dir', fallback='logs/')
        train_results.to_csv(log_path+'repair_model_train_results.csv', index=True)

        # 保存模型参数
        save_path = self.config.get('repair_model', 'save_path', fallback=None)
        if save_path is not None:
            self.repair_module.save_model(save_path)

    def predict(self, data):
        if isinstance(data, pd.Series):
            data = pd.DataFrame([data])

        indata = self.get_tensor_data(data)
        labels = self.detect_module.predict(indata)

        return labels

    def infer(self, data):
        if isinstance(data, pd.Series):
            data = pd.DataFrame([data])

        data_corrupt = copy.deepcopy(data)
        for index, row_data in data_corrupt.iterrows():
            fmask = self.infer_module.get_feature_label(row_data)
            row_data[fmask] = np.nan

        return data_corrupt, data_corrupt.isnull()

    def impute(self, data_corrupt):
        if isinstance(data_corrupt, pd.Series):
            data_corrupt = pd.DataFrame([data_corrupt])

        fmask = data_corrupt.isnull()

        data_scaled = dtsm.get_normalized_data(data_corrupt)
        data_scaled = data_scaled.replace(np.nan, 0)
        indata = self.get_tensor_data(data_scaled)
        outdata = self.repair_module.impute(indata)

        data_array = outdata.detach().numpy()
        data_array = data_array.reshape((-1, 5))
        data_df = pd.DataFrame(data_array, 
                               columns=data_corrupt.columns, 
                               index=data_corrupt.index)
        data_recon = dtsm.get_denormalized_data(data_df)

        data_impute = copy.deepcopy(data_corrupt)
        data_impute[fmask] = data_recon[fmask]

        return data_impute

    def detect(self, single_data_full):
        if isinstance(single_data_full, pd.Series):
            data_full = pd.DataFrame([single_data_full])
        else:
            data_full = single_data_full

        if data_full.shape[0] != 1:
            raise ValueError('Not support multiple records now, please input the data only contains one records')

        if data_full.shape[1] > 5:
            data_filtered = data_full.reindex(columns=dtsm.FEATURE_COL)
        else:
            data_filtered = data_full

        labels = self.predict(data_filtered)
        is_fault = bool(labels[0])

        return is_fault

    def repair(self, single_data_full, iters=10):
        if isinstance(single_data_full, pd.Series):
            data_full = pd.DataFrame([single_data_full])
        else:
            data_full = single_data_full

        if data_full.shape[0] != 1:
            raise ValueError('Not support multiple records now, please input the data only contains one records')

        if data_full.shape[1] > 5:
            data_filtered = data_full.reindex(columns=dtsm.FEATURE_COL)
        else:
            data_filtered = data_full

        # 修复故障数据直到检测为正常数据为止，或者当迭代次数达到最大才停止
        indata = data_filtered
        for i in range(iters):
            data_corrupt, fmask = self.infer(indata)
            data_impute = self.impute(data_corrupt)
            label = self.predict(data_impute)[0]
            if label == 0:
                break

            indata = data_impute

        # 构建修复后的完整数据
        data_impute_full = copy.deepcopy(data_full)
        for fname in data_impute.columns:
            data_impute_full[fname] = data_impute[fname]

        return bool(label), data_impute_full

