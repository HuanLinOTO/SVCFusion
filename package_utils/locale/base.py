locale_name = ""  # zh-cn
locale_display_name = ""  # 简体中文
class Locale:
    unknown_model_type_tip = ""  # 模型类型未知，请手动选择
    preprocess_failed_tip = ""  # 预处理失败！请截图控制台信息并加群反馈
    class device_chooser:
        device_dropdown_label = ""  # 设备
    class model_chooser:
        submit_btn_value = ""  # 选择模型
        model_type_dropdown_label = ""  # 模型类型
        search_path_label = ""  # 搜索路径
        workdir_name = ""  # 工作目录
        archieve_dir_name = ""  # 已归档训练
        models_dir_name = ""  # models 文件夹
        no_model_value = ""  # 无模型
        unuse_value = ""  # 不使用
        no_spk_value = ""  # 无说话人
        choose_model_dropdown_prefix = ""  # 选择模型
        refresh_btn_value = ""  # 刷新选项
        spk_dropdown_label = ""  # 选择说话人
        no_spk_option = ""  # 未加载模型
    class form:
        submit_btn_value = ""  # 提交
        audio_output_1 = ""  # 输出结果
        audio_output_2 = ""  # 输出结果/伴奏
        textbox_output = ""  # 输出结果
        dorpdown_liked_checkbox_yes = ""  # 是
        dorpdown_liked_checkbox_no = ""  # 否
    class model_manager:
        pack_btn_value = ""  # 打包模型
        pack_result_label = ""  # 打包结果
        packing_tip = ""  # 正在打包，请勿多次点击
        unpackable_tip = ""  # 该模型不支持打包
        clean_log_btn_value = ""  # 清空日志(确认不再训练再清空)
        change_model_type_info = ""  # 
        change_model_type_btn_value = ""  # 确认更改
        change_success_tip = ""  # 更改成功
        change_fail_tip = ""  # 更改失败
    class main_ui:
        release_memory_btn_value = ""  # 尝试释放显存/内存
        released_tip = ""  # 已尝试释放显存/内存
        infer_tab = ""  # 💡推理
        preprocess_tab = ""  # ⏳数据处理
        train_tab = ""  # 🏋️‍♂️训练
        tools_tab = ""  # 🛠️小工具
        settings_tab = ""  # 🪡设置
        model_manager_tab = ""  # 模型管理
        install_model_tab = ""  # 安装模型
        fish_audio_preprocess_tab = ""  # 简单音频处理
        vocal_remove_tab = ""  # 人声分离
        detect_spk_tip = ""  # 已检测到的角色：
        spk_not_found_tip = ""  # 未检测到任何角色
    class preprocess:
        tip = ""  # 
        little_vram_tip = ""  # 
        open_dataset_folder_btn_value = ""  # 打开数据集文件夹
        choose_model_label = ""  # 选择模型
    class train:
        current_train_model_label = ""  # 当前训练模型
        fouzu_tip = ""  # ~~整了个赛博佛祖，希望对你有帮助~~
        gd_plus_1 = ""  # 点我加功德
        gd_plus_1_tip = ""  # 功德 +1，炸炉 -1
        choose_sub_model_label = ""  # 选择子模型
        archieve_btn_value = ""  # 归档工作目录
        stop_btn_value = ""  # 停止训练
        archieving_tip = ""  # 正在归档，请勿多次点击
        archieved_tip = ""  # 归档完成，请查看打开的文件夹
        stopped_tip = ""  # 已发送停止训练命令，请查看训练窗口
        tensorboard_btn = ""  # 启动 Tensorboard
        launching_tb_tip = ""  # 正在启动 Tensorboard，请稍后
        launched_tb_tip = ""  # Tensorboard 已在 {1} 开放
    class settings:
        page = ""  # 页面
        pkg_settings_label = ""  # 整合包设置
        sovits_settings_label = ""  # So-VITS-SVC 设置
        ddsp6_settings_label = ""  # DDSP-SVC 6 设置
        class pkg:
            lang_label = ""  # 语言
            lang_info = ""  # 更改语言需要重启整合包
        class sovits:
            resolve_port_clash_label = ""  # 尝试解决端口冲突问题
        class ddsp6:
            pretrained_model_preference_dropdown_label = ""  # 底模偏好
            default_pretrained_model = ""  # 默认底模 512 x 6
            large_pretrained_model = ""  # 大网络底模 1024 x 12
        saved_tip = ""  # 已保存
    class install_model:
        tip = ""  # 
        file_label = ""  # 上传模型包
        model_name_label = ""  # 模型名称
        model_name_placeholder = ""  # 请输入模型名称
        submit_btn_value = ""  # 安装模型
    class path_chooser:
        input_path_label = ""  # 输入文件夹
        output_path_label = ""  # 输出文件夹
    class fish_audio_preprocess:
        to_wav_tab = ""  # 批量转 WAV
        slice_audio_tab = ""  # 切音机
        preprocess_tab = ""  # 数据处理
        max_duration_label = ""  # 最大时长
        submit_btn_value = ""  # 开始
        input_path_not_exist_tip = ""  # 输入路径不存在
    class vocal_remove:
        input_audio_label = ""  # 输入音频
        submit_btn_value = ""  # 开始
        vocal_label = ""  # 输出-人声
        inst_label = ""  # 输出-伴奏
    class common_infer:
        audio_label = ""  # 音频文件
        use_vocal_remove_label = ""  # 去除伴奏
        use_vocal_remove_info = ""  # 是否去除伴奏
        use_harmony_remove_label = ""  # 去除和声
        use_harmony_remove_info = ""  # 是否去除和声（必须先勾选移除伴奏）
        f0_label = ""  # f0 提取器
        f0_info = ""  # 用于音高提取/预测的模型
        keychange_label = ""  # 变调
        keychange_info = ""  # 参考：男转女 12，女转男 -12，音色不像可以调节这个
        threshold_label = ""  # 切片阈值
        threshold_info = ""  # 人声切片的阈值，如果有底噪可以调为 -40 或更高
    class diff_based_infer:
        method_label = ""  # 采样器
        method_info = ""  # 用于 reflow 的采样器
        infer_step_label = ""  # 推理步数
        infer_step_info = ""  # 推理步长，默认就行
        t_start_label = ""  # T Start
        t_start_info = ""  # 不知道
    class diff_based_preprocess:
        method_label = ""  # f0 提取器
        method_info = ""  # 用于 reflow 的采样器
    class common_preprocess:
        encoder_label = ""  # 声音编码器
        encoder_info = ""  # 用于对声音进行编码的模型
        f0_label = ""  # f0 提取器
        f0_info = ""  # 用于音高提取/预测的模型
    class sovits:
        dataset_not_complete_tip = ""  # 数据集不完整，请检查数据集或重新预处理
        finished = ""  # 完成
        class train_main:
            log_interval_label = ""  # 日志间隔
            log_interval_info = ""  # 每 N 步输出一次日志
            eval_interval_label = ""  # 验证间隔
            eval_interval_info = ""  # 每 N 步保存一次并验证
            all_in_mem_label = ""  # 缓存全数据集
            all_in_mem_info = ""  # (
            keep_ckpts_label = ""  # 保留检查点
            keep_ckpts_info = ""  # 保留最近 N 个检查点
            batch_size_label = ""  # 训练批次大小
            batch_size_info = ""  # 越大越好，越大越占显存
            learning_rate_label = ""  # 学习率
            learning_rate_info = ""  # 学习率
            num_workers_label = ""  # 数据加载器进程数
            num_workers_info = ""  # 仅在 CPU 核心数大于 4 时启用，遵循大就是好原则
            half_type_label = ""  # 精度
            half_type_info = ""  # 选择 fp16 可以获得更快的速度，但是炸炉概率 up up
        class train_diff:
            batchsize_label = ""  # 训练批次大小
            batchsize_info = ""  # 越大越好，越大越占显存，注意不能超过训练集条数
            num_workers_label = ""  # 训练进程数
            num_workers_info = ""  # 如果你显卡挺好，可以设为 0
            amp_dtype_label = ""  # 训练精度
            amp_dtype_info = ""  # 选择 fp16、bf16 可以获得更快的速度，但是炸炉概率 up up
            lr_label = ""  # 学习率
            lr_info = ""  # 不建议动
            interval_val_label = ""  # 验证间隔
            interval_val_info = ""  # 每 N 步验证一次，同时保存
            interval_log_label = ""  # 日志间隔
            interval_log_info = ""  # 每 N 步输出一次日志
            interval_force_save_label = ""  # 强制保存模型间隔
            interval_force_save_info = ""  # 每 N 步保存一次模型
            gamma_label = ""  # lr 衰减力度
            gamma_info = ""  # 不建议动
            cache_device_label = ""  # 缓存设备
            cache_device_info = ""  # 选择 cuda 可以获得更快的速度，但是需要更大显存的显卡 (SoVITS 主模型无效)
            cache_all_data_label = ""  # 缓存所有数据
            cache_all_data_info = ""  # 可以获得更快的速度，但是需要大内存/显存的设备
            epochs_label = ""  # 最大训练轮数
            epochs_info = ""  # 达到设定值时将会停止训练
            use_pretrain_label = ""  # 使用预训练模型
            use_pretrain_info = ""  # 勾选可以大幅减少训练时间，如果你不懂，不要动
        class train_cluster:
            cluster_or_index_label = ""  # 聚类或检索
            cluster_or_index_info = ""  # 要训练聚类还是检索模型，检索咬字比聚类稍好
            use_gpu_label = ""  # 使用 GPU
            use_gpu_info = ""  # 使用 GPU 可以加速训练，该参数只聚类可用
        class infer:
            cluster_infer_ratio_label = ""  # 聚类/特征比例
            cluster_infer_ratio_info = ""  # (
            linear_gradient_info = ""  # 两段音频切片的交叉淡入长度
            linear_gradient_label = ""  # 渐变长度
            k_step_label = ""  # 扩散步数
            k_step_info = ""  # 越大越接近扩散模型的结果，默认100
            enhancer_adaptive_key_label = ""  # 增强器适应
            enhancer_adaptive_key_info = ""  # 使增强器适应更高的音域(单位为半音数)|默认为0
            f0_filter_threshold_label = ""  # f0 过滤阈值
            f0_filter_threshold_info = ""  # 只有使用crepe时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音
            audio_predict_f0_label = ""  # 自动 f0 预测
            audio_predict_f0_info = ""  # (
            second_encoding_label = ""  # 二次编码
            second_encoding_info = ""  # (
            clip_label = ""  # 强制切片长度
            clip_info = ""  # 强制音频切片长度, 0 为不强制
        class preprocess:
            use_diff_label = ""  # 训练浅扩散
            use_diff_info = ""  # 勾选后将会生成训练浅扩散需要的文件，会比不选慢
            vol_aug_label = ""  # 响度嵌入
            vol_aug_info = ""  # 勾选后将会使用响度嵌入
            num_workers_label = ""  # 进程数
            num_workers_info = ""  # 理论越大越快
            subprocess_num_workers_label = ""  # 每个进程的线程数
            subprocess_num_workers_info = ""  # 理论越大越快
            debug_label = ""  # 是否开启 Debug 模式
            debug_info = ""  # 开启后会输出调试信息，非特殊情况没必要开
        class model_types:
            main = ""  # 主模型
            diff = ""  # 浅扩散
            cluster = ""  # 聚类/检索模型
        class model_chooser_extra:
            enhance_label = ""  # NSFHifigan 音频增强
            enhance_info = ""  # (
            feature_retrieval_label = ""  # 启用特征提取
            feature_retrieval_info = ""  # 是否使用特征检索，如果使用聚类模型将被禁用
    class ddsp6:
        infer_tip = ""  # 推理 DDSP 模型
        class model_types:
            cascade = ""  # 级联模型
        class train:
            batch_size_label = ""  # 训练批次大小
            batch_size_info = ""  # 越大越好，越大越占显存，注意不能超过训练集条数
            num_workers_label = ""  # 训练进程数
            num_workers_info = ""  # 如果你显卡挺好，可以设为 0
            amp_dtype_label = ""  # 训练精度
            amp_dtype_info = ""  # 选择 fp16、bf16 可以获得更快的速度，但是炸炉概率 up up
            lr_label = ""  # 学习率
            lr_info = ""  # 不建议动
            interval_val_label = ""  # 验证间隔
            interval_val_info = ""  # 每 N 步验证一次，同时保存
            interval_log_label = ""  # 日志间隔
            interval_log_info = ""  # 每 N 步输出一次日志
            interval_force_save_label = ""  # 强制保存模型间隔
            interval_force_save_info = ""  # 每 N 步保存一次模型
            gamma_label = ""  # lr 衰减力度
            gamma_info = ""  # 不建议动
            cache_device_label = ""  # 缓存设备
            cache_device_info = ""  # 选择 cuda 可以获得更快的速度，但是需要更大显存的显卡 (SoVITS 主模型无效)
            cache_all_data_label = ""  # 缓存所有数据
            cache_all_data_info = ""  # 可以获得更快的速度，但是需要大内存/显存的设备
            epochs_label = ""  # 最大训练轮数
            epochs_info = ""  # 达到设定值时将会停止训练
            use_pretrain_label = ""  # 使用预训练模型
            use_pretrain_info = ""  # 勾选可以大幅减少训练时间，如果你不懂，不要动
    class reflow:
        infer_tip = ""  # 推理 ReflowVAESVC 模型
        class train:
            batch_size_label = ""  # 训练批次大小
            batch_size_info = ""  # 越大越好，越大越占显存，注意不能超过训练集条数
            num_workers_label = ""  # 训练进程数
            num_workers_info = ""  # 如果你显卡挺好，可以设为 0
            amp_dtype_label = ""  # 训练精度
            amp_dtype_info = ""  # 选择 fp16、bf16 可以获得更快的速度，但是炸炉概率 up up
            lr_label = ""  # 学习率
            lr_info = ""  # 不建议动
            interval_val_label = ""  # 验证间隔
            interval_val_info = ""  # 每 N 步验证一次，同时保存
            interval_log_label = ""  # 日志间隔
            interval_log_info = ""  # 每 N 步输出一次日志
            interval_force_save_label = ""  # 强制保存模型间隔
            interval_force_save_info = ""  # 每 N 步保存一次模型
            gamma_label = ""  # lr 衰减力度
            gamma_info = ""  # 不建议动
            cache_device_label = ""  # 缓存设备
            cache_device_info = ""  # 选择 cuda 可以获得更快的速度，但是需要更大显存的显卡 (SoVITS 主模型无效)
            cache_all_data_label = ""  # 缓存所有数据
            cache_all_data_info = ""  # 可以获得更快的速度，但是需要大内存/显存的设备
            epochs_label = ""  # 最大训练轮数
            epochs_info = ""  # 达到设定值时将会停止训练
            use_pretrain_label = ""  # 使用预训练模型
            use_pretrain_info = ""  # 勾选可以大幅减少训练时间，如果你不懂，不要动
        class model_types:
            cascade = ""  # 级联模型
    default_spk_name = ""  # 默认说话人
    preprocess_draw_desc = ""  # 划分验证集
    preprocess_desc = ""  # 预处理(进度去终端看)
    preprocess_finished = ""  # 预处理完成
