# oneDNN

## 介绍

[oneDNN](https://github.com/oneapi-src/oneDNN) 是 Intel 开源的跨平台深度学习性能加速库，通过[文档](https://oneapi-src.github.io/oneDNN/)可以了解到被支持的原语，DeepRec 中已经加入了 oneDNN 的支持，只需要在 DeepRec 编译命令中加入关于 oneDNN 的编译选项：`--config=mkl_threadpool --copt=-march=skylake-avx512` 即可开启 oneDNN 加速算子计算。

Tips: MKL-DNN 被重命名为 DNNL，之后又被重命名为 oneDNN；TensorFlow 初期采用的是 MKL 加速算子计算，在之后的版本迭代中，逐步使用 oneDNN 替换了 MKL，但宏定义还是仍然保留。



DeepRec 关于 oneDNN 的宏定义：

| 宏定义                           | 可设值（默认值加粗）                           | 解释                                                         |
| :------------------------------- | ---------------------------------------------- | ------------------------------------------------------------ |
| TF_MKL_PRIMITIVE_ONLY_FOR_RECO   | **1/true**, 0/false                            | 1: 仅替换推荐模型中oneDNN支持的算子；0: 替换成所有oneDNN支持的的算子 |
| TF_MKL_OPTIMIZE_PRIMITIVE_MEMUSE | **1/true**,  0/false                           | 1: 通过释放原语来减少内存，再次使用会重建；0: 不释放原语     |
| TF_DISABLE_MKL                   | **0**, 1                                       | 0: Enable MKL; 1: Disable MKL                                |
| TF_MKL_NUM_INTRAOP               | 整数值，如14，**默认不设置**                   | 整数值：设置 oneDNN 使用的 intra 线程数；不设置：使用最多的 TF intra 线程数 |
| ONEDNN_VERBOSE                   | **0**/1/2                                      | 打印 oneDNN 原语输出的 log 的[等级](https://oneapi-src.github.io/oneDNN/dev_guide_verbose.html) |
| DNNL_MAX_CPU_ISA                 | **ALL**, AVX512_CORE_AMX,  AVX512_CORE_BF16, … | 指定 oneDNN(版本小于2.5.0时) 使用的[最高指令集](https://oneapi-src.github.io/oneDNN/v2.4/dev_guide_cpu_dispatcher_control.html#run-time-controls) |
| ONEDNN_MAX_CPU_ISA               | **ALL**, AVX512_CORE_AMX,  AVX512_CORE_BF16, … | 指定 oneDNN(版本大于等于2.5.0时) 使用的[最高指令集](https://oneapi-src.github.io/oneDNN/dev_guide_cpu_dispatcher_control.html) |



oneDNN 支持的原语：

| 原语                                   | 支持的类型                  | 支持的后向操作                    |
| :------------------------------------- | :-------------------------- | --------------------------------- |
| Matrix Multiplication                  | f32, bf16, f16, u8, s8      | Scale, Zero, Eltwise, Sum, Binary |
| Inner Product                          | f32, bf16, f16, u8, s8      | Scale, Eltwise, Sum, Binary       |
| Layer Normalization                    | f32, bf16, f16              | /                                 |
| Batch Normalization                    | f32, bf16, f16, s8          | Eltwise                           |
| Local Response Normalization (LRN)     | f32, bf16, f16              | /                                 |
| Binary (+, =, *, /, >, <, min, max...) | f32, bf16, f16, u8, s8      | Scale, Eltwise, Sum, Binary       |
| Eltwise (relu, gelu, tanh, linear...)  | f32, s32, bf16, f16, u8, s8 | Binary                            |
| PReLU                                  | f32, s32, bf16, s8, u8      | /                                 |
| Sum                                    | f32, s32, bf16, f16, u8, s8 | /                                 |
| Reduction                              | f32, bf16, u8, s8           | Eltwise, Sum, Binary              |
| Softmax                                | f32, bf16, f16              | /                                 |
| LogSoftmax                             | f32, bf16                   | /                                 |
| Reorder                                | f32, s32, bf16, f16, u8, s8 | Scale, Sum                        |
| Concat                                 | f32, s32, bf16, f16, u8, s8 | /                                 |
| Convolution                            | f32, bf16, f16, u8, s8      | Scale, Zero, Eltwise, Sum, Binary |
| Pooling                                | f32, s32, bf16, f16, u8, s8 | Binary                            |
| RNN (LSTM, GRU, Vanilla RNN...)        | f32, bf16, f16, u8, s8      | /                                 |
| Resampling                             | f32, s32, bf16, f16, s8, u8 | Eltwise, Sum, Binary              |
| Shuffle                                | f32, s32, bf16, s8, u8      | /                                 |



## BFloat16

BFloat16(BF16)，是 Intel 在 Cooper Lake ([阿里云hfg7规格族](https://help.aliyun.com/document_detail/25378.html?spm=5176.2020520101.vmBInfo.instanceType.4a944df5PvCcED#hfg7)) 及其之后处理器上推出的一种计算格式，用于加速深度学习的训练和推理。其与其他常用数据格式的比较：

![img_1.png](oneDNN/BF16.png)

#### 使用条件与方法

使用条件：Intel Xeon Cooper Lake CPU ([阿里云hfg7规格族](https://help.aliyun.com/document_detail/25378.html?spm=5176.2020520101.vmBInfo.instanceType.4a944df5PvCcED#hfg7))

使用方法：由于推荐场景对模型精度的要求极其严苛，所以为了提升模型性能的同时，兼顾模型精度，用户可以通过以下方式，自由控制 BF16 计算图。

- 步骤 1：在 `tf.variable_scope(…)` 之后添加 `.keep_weights(dtype=tf.float32)`，用于保持当前权重为 FP32 类型；

- 步骤 2：添加 `tf.cast(…, dtype=tf.bfloat16)` 将 inputs tensor 转换为 BF16 类型；

- 步骤 3：添加 `tf.cast(…, dtype=tf.float32)` 将 outputs tensor 转换为 FP32 类型。

```
with tf.variable_scope(…).keep_weights(dtype=tf.float32):
  inputs_bf16 = tf.cast(inputs, dtype=tf.bfloat16)
  … // BF16 graph, FP32 weights
  outputs = tf.cast(outputs_bf16, dtype=tf.float32)
```

代码示例：
```
import tensorflow as tf

inputs = tf.ones([4, 8], tf.float32)

with tf.variable_scope('dnn', reuse=tf.AUTO_REUSE).keep_weights(dtype=tf.float32):
  # cast inputs to BF16
  inputs = tf.cast(inputs, dtype=tf.bfloat16)
  outputs = tf.layers.dense(inputs, units=8, activation=tf.nn.relu)
  outputs = tf.layers.dense(inputs, units=1, activation=None)
  # cast ouputs to FP32
  outputs = tf.cast(outputs, dtype=tf.float32)

  outputs = tf.nn.softmax(outputs)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print(sess.run(outputs))
```

为兼顾模型精度，DeepRec在variable_scope中提供了`keep_weights(dtype=dtypes.float32)`方法，使用该方法后，该变量域中所有变量将会以FP32的格式保存，在图中自动添加cast操作，将其转换为BF16格式进行计算。  
由于引入的cast操作会带来额外的计算开销，为了减小这部分开销，DeepRec会自动将cast算子和就近的算子进行融合，提高运行速度。  
DeepRec将进行下列cast相关算子的融合操作
- MatMul + Cast
- Concat + Cast
- Split + Cast


#### 性能对比
使用Modelzoo中模型，对比DeepRec开启BF16后对比FP32的性能提升。Modelzoo中模型通过添加`--bf16`参数启用BF16特性。  
测试机器使用阿里云ECS云服务器，Intel Xeon Cooper Lake CPU，规格为[ecs.hfg7.2xlarge](https://help.aliyun.com/document_detail/25378.html?spm=5176.2020520101.vmBInfo.instanceType.4a944df5PvCcED#hfg7)
- 硬件配置：
  - Intel(R) Xeon(R) Platinum 8369HC CPU @ 3.30GHz
  - CPU(s): 8
  - Socket(s): 1
  - Core(s) per socket: 4
  - Thread(s) per core: 2
  - Memory: 32G
- 软件配置
  - kernel: 4.18.0-348.2.1.el8_5.x86_64
  - OS: CentOS Linux release 8.5.2111
  - GCC: 8.5.0
  - Docker: 20.10.12
  - Python: 3.6.8

性能结果：
| **Throughput** | **WDL**  | **DeepFM** | **DSSM**  |
|----------------|----------|------------|-----------|
| FP32           | 15792.49 | 30718.6    | 114436.87 |
| FP32+BF16      | 22633.8  | 34554.67   | 125995.83 |
| Speedup        | 1.43x    | 1.12x      | 1.10x     |

