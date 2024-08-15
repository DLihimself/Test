# SignalDetectRepairSystem

A system to simulate signal data and train the detect module and repair module

光信号系统，提供以下主要功能：
1. 检查光信号是否为故障信号
2. 诊断引起故障的特征字段
2. 修复故障信号

光信号系统由以下3个主要模块组成：
1. 检查模块（CNN-1D分类器）：检测光信号是否为故障信号
2. 诊断模块：诊断哪些特征值可能引起故障，并会将故障特征设置为空值
3. 修复模块（VAE+CNN）：修复带有空值的故障信息

启动http服务的命令是：
`
python http_server.py
`

系统支持启动http服务器
### 1. 训练模型
* 请求地址: http://ip:port/train
* 请求方法: GET
* 请求参数: 无
* 返回: 200 （无其他数据）


### 2. 检测信号是否故障
* 请求地址: http://ip:port/detect
* 请求方法: GET
* 请求url的例子：http://127.0.0.1:8000/detect?signalStrength=0.936055&distance=103.333769&frequency=204.650447&temperature=-9.186339&humidity=98.182911&fiberType=SMF&encoding=DM&wavelength=1465914.218

* 请求参数: 
<table>
  <tr>
    <th>参数</th>
    <th>取值范围</th>
  </tr>
  <tr>
    <td>signalStrength</td>
    <td>[0, 1]</td>
  </tr>
  <tr>
    <td>distance</td>
    <td>[10, 120]</td>
  </tr>
  <tr>
    <td>fiberType</td>
    <td>['SMF', 'MMF', 'PMF', 'DSF', 'AGF']</td>
  </tr>
  <tr>
    <td>frequency</td>
    <td>[10, 800]</td>
  </tr>
  <tr>
    <td>temperature</td>
    <td>[-40, 85]</td>
  </tr>
  <tr>
    <td>humidity</td>
    <td>[0, 100]</td>
  </tr>
  <tr>
    <td>encoding</td>
    <td>['DM', 'EM', 'CM', 'DSSS']</td>
  </tr>
  <tr>
    <td>wavelength</td>
    <td>根据frequency取值</td>
  </tr>
</table>  
注：createDtm字段不需要提供，系统会根据请求时间自动创建

* 返回:
<table>
  <tr>
    <th>HTTP code</th>
    <th>说明</th>
  </tr>
  <tr>
    <td>200</td>
    <td>response是{"is_fault": true}</td>
  </tr>
  <tr>
    <td>400</td>
    <td>如果有字段缺失，或者取值范围不对会返回400，并提供相应错误信息</td>
  </tr>
</table>  

### 3. 修复故障数据
* 请求地址: http://ip:port/repair
* 请求方法: GET
* 请求url的例子：http://127.0.0.1:8000/repair?signalStrength=0.936055&distance=103.333769&frequency=204.650447&temperature=-9.186339&humidity=98.182911&fiberType=SMF&encoding=DM&wavelength=1465914.218
* 请求参数: 同上（detect接口）
* 返回:
<table>
  <tr>
    <th>HTTP code</th>
    <th>说明</th>
  </tr>
  <tr>
    <td>200</td>
    <td>response是{"is_fault": false, "data_impute": {"signalStrength": 0.936055, "distance": 16.887174606323242, "fiberType": "SMF", "frequency": 204.650447, "temperature": 24.783309936523438, "humidity": 7.330900192260742, "encoding": "DM", "wavelength": 1465914.218, "createDtm": "2024-02-05 19:00:16.127096"}}</td>
  </tr>
  <tr>
    <td>400</td>
    <td>如果有字段缺失，或者取值范围不对会返回400，并提供相应错误信息</td>
  </tr>
</table>  
注：返回数据中"is_fault"表示修复数据输入到检查模块的识别结果，"data_impute"返回修复后的数据



### 4. 文件功能说明：
* config/setup.ini：系统配置文件
* logs:系统日志
* modules/detect_module.py: 故障检测模型
* modules/repair_module.py: 数据修复模型
* modules/models.py:定义所有网络模型，包括故障检测模块的一维CNN网络，数据修复模块的VAE模型
* modules/inference_module:故障诊断模块（就是诊断哪些特征值会引起故障）
* store/data/train_data.csv: 训练数据集
* store/data/test_data.csv: 测试数据集
* store/models/repair_model: VAE模型保存或加载网络参数的目录（VAE网络需要分别加载encoder和decoder, 所以目录里有两个文件encoder.pth和decoder.pth）
* store/models/detect_model: 检测模块中一维CNN网络保存或加载的网络参数
* http_server.py:向外界提供接口服务，使用python HTTPServer框架
* signal_system.py:最核心的文件，用于封装电信号故障检测、故障诊断、数据修复等功能

### 5. 流程解释：
整个的流程是：
1. 通过训练数据集，训练模型，会生成： 故障检测模型 和  数据修复模型
2. 通过故障检测模型用于检测一条光纤信号是否存在故障（判断一个人是否有疾病）
3. 通过诊断模型用于确定这一条记录中 哪些字段可能引起故障（判断一个人是身体的那个部位引发的疾病）
4. 通过 数据修复模型，将 引起故障的各个数据字段进行数据修复（通过开药方去自疗这个疾病）
5. 把修复之后的那一条记录，通过 故障检测模型去检测，发现该条记录已经没有故障（服药之后，再去医院复查，发生疾病消失）