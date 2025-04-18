from SVCFusion.locale.base import Locale

locale_name = "zh-cn"
locale_display_name = "简体中文"


class _Locale(Locale):
    unknown_model_type_tip = "模型类型未知，请去 小工具-模型管理 确认模型类型"
    preprocess_failed_tip = "预处理失败！请截图控制台信息并加群反馈"

    error_when_infer = "推理时遇到错误<br>已跳过{1}文件<br>详细查看控制台<br>{2}"

    class device_chooser(Locale.device_chooser):
        device_dropdown_label = "设备"

    class model_chooser(Locale.model_chooser):
        submit_btn_value = "选择模型"
        model_type_dropdown_label = "模型类型"
        search_path_label = "搜索路径"

        workdir_name = "工作目录"
        archive_dir_name = "已归档训练"
        models_dir_name = "models 文件夹"

        no_model_value = "无模型"
        unuse_value = "不使用"
        no_spk_value = "无说话人"

        choose_model_dropdown_prefix = "选择模型"

        refresh_btn_value = "刷新选项"

        spk_dropdown_label = "选择说话人"
        no_spk_option = "未加载模型"

    class form(Locale.form):
        submit_btn_value = "提交"
        audio_output_1 = "输出结果"
        audio_output_2 = "输出结果/伴奏"
        audio_output_3 = "输出结果/混音"
        textbox_output = "输出结果"

        dorpdown_liked_checkbox_yes = "是"
        dorpdown_liked_checkbox_no = "否"

        cancel_btn_value = "取消"
        canceling_tip = "正在取消，请稍后"

    class model_manager(Locale.model_manager):
        choose_model_title = "选择模型"
        action_title = "操作"

        pack_btn_value = "打包模型"
        pack_result_label = "打包结果"
        packing_tip = "正在打包，请勿多次点击"
        unpackable_tip = "该模型不支持打包"

        clean_log_btn_value = "清空日志(确认不再训练再清空)"

        change_model_type_info = """
        #### 更改模型类型
        仅在发生无法识别模型类型时使用！不是转换模型类型！是更改识别的模型类型！
        """
        change_model_type_btn_value = "确认更改"
        change_success_tip = "更改成功"
        change_fail_tip = "更改失败"

        move_folder_tip = "#### 移动到 models 目录"
        move_folder_name = "模型名称"
        move_folder_name_auto_get = "自动获取"
        move_folder_btn_value = "移动"
        other_text = "等"
        moving_tip = "正在移动，请勿多次点击"
        moved_tip = "已移动到 {1}，刷新后可用"

    class main_ui(Locale.main_ui):
        release_memory_btn_value = "尝试释放显存/内存"
        released_tip = "已尝试释放显存/内存"
        infer_tab = "💡推理"
        preprocess_tab = "⏳数据处理"
        train_tab = "🏋️‍♂️训练"
        tools_tab = "🛠️小工具"
        settings_tab = "⚙️设置"

        model_tools_tab = "模型相关"
        audio_tools_tab = "音频相关"
        realtime_tools_tab = "实时"
        dlc_install_tools_tab = "DLC"

        start_ddsp_realtime_gui_btn = "启动 DDSP-SVC 6.0 实时 GUI"
        start_ddsp6_1_realtime_gui_btn = "启动 DDSP-SVC 6.1 实时 GUI"

        starting_tip = "正在启动，请稍后，不要重复点击，后果很严重"

        load_model_btn_value = "加载模型"
        infer_btn_value = "开始推理"

        model_manager_tab = "模型管理"
        install_model_tab = "安装模型"
        fish_audio_preprocess_tab = "简单音频处理"
        vocal_separation_tab = "人声分离"
        compatible_tab = "模型兼容"

        move_folder_name_error = "模型名称不合法"

        detect_spk_tip = "已检测到的角色："
        spk_not_found_tip = "未检测到任何角色"

    class DLC(Locale.DLC):
        dlc_install_label = "上传新 DLC"
        dlc_install_btn_value = "安装 DLC"
        dlc_installing_tip = "正在安装"
        dlc_install_success = "安装成功"
        dlc_install_failed = "安装失败"
        dlc_install_empty = "未选择文件"
        dlc_install_ext_error = "不支持非 .sf_dlc 文件格式"

    class compatible_models(Locale.compatible_models):
        upload_error = "上传错误，请检查文件是否完整"
        model_name_label = "模型名称"
        upload_success = "上传成功"
        model_exists = "模型已存在"

        compatible_sovits = "SoVITS 模型兼容"
        sovits_main_model_label = "SoVITS 主模型"
        sovits_diff_model_label = "SoVITS 浅扩散"
        sovits_cluster_model_label = "SoVITS 聚类/检索"

        sovits_main_model_config_label = "SoVITS 主模型配置"
        sovits_diff_model_config_label = "SoVITS 浅扩散配置"

    class preprocess(Locale.preprocess):
        tip = """
            请先把你的数据集（也就是一堆 `.wav` 文件）放到整合包下的 `dataset_raw/你的角色名字` 文件夹中

            你可以通过新建多个角色文件夹的方式同时训练多个角色

            放置完毕后你的目录应该长这样

            ```
            dataset_raw/
            |-你的角色名字1/
            |  | 1.wav
            |  | 2.wav
            |  | 3.wav
            |  ...
            |-你的角色名字2/
            |  | 1.wav
            |  | 2.wav
            |  | 3.wav
            |  ...
            ```

            如果你啥也不懂，直接点击下面的按钮进行全自动数据处理

            如果你懂得参数的意义，可以调为手动模式，进行更加详细的数据处理
            
            **CPU 用户请使用 FCPE 作为 F0 提取器/预测器**
        """
        low_vram_tip = """
            ## 当前设备没有一张显卡显存大于 6GB，仅推荐训练 DDSP 模型
            
            注意 这并不代表你不能训练！！
        """

        open_dataset_folder_btn_value = "打开数据集文件夹"

        choose_model_label = "选择模型"
        start_preprocess_btn_value = "开始预处理"

    class train(Locale.train):
        current_train_model_label = "当前训练模型"

        fouzu_tip = "~~整了个赛博佛祖，希望对你有帮助~~"

        gd_plus_1 = "点我加功德"
        gd_plus_1_tip = "功德 +1，炸炉 -1"

        choose_sub_model_label = "选择子模型"
        choose_pretrain_model_label = "选择预训练模型"
        choose_pretrain_model_info = (
            "根据你的设备选择合适的底模，你可以在官方网站上获取更多底模"
        )
        pretrain_model_not_found_tip = "未找到预训练模型"

        pretrain_model_vec = "编码器"
        pretrain_model_vocoder = "声码器"
        pretrain_model_size = "网络参数"
        pretrain_model_attn = "是否有注意力机制"
        official_pretrain_model = "官方预训练模型"

        load_pretrained_failed_tip = (
            "加载预训练模型失败，可能为选择的模型不适配当前参数或无预训练模型"
        )

        start_train_btn_value = "开始/继续训练"

        archive_btn_value = "归档工作目录"
        stop_btn_value = "停止训练"

        archieving_tip = "正在归档，请勿多次点击"
        archived_tip = "归档完成，请查看打开的文件夹"

        stopped_tip = "已发送停止训练命令，请查看训练窗口"

        tensorboard_btn = "启动 Tensorboard"

        launching_tb_tip = "正在启动 Tensorboard，请稍后"
        launched_tb_tip = "Tensorboard 已在 {1} 开放"

    class settings(Locale.settings):
        page = "页面"

        save_btn_value = "保存设置"

        pkg_settings_label = "整合包设置"
        infer_settings_label = "推理设置"
        sovits_settings_label = "So-VITS-SVC 设置"
        ddsp6_settings_label = "DDSP-SVC 6 设置"
        ddsp6_1_settings_label = "DDSP-SVC 6.1 设置"

        class pkg(Locale.settings.pkg):
            lang_label = "语言"
            lang_info = "更改语言需要重启整合包"

        class infer(Locale.settings.infer):
            msst_device_label = "运行分离任务使用设备"

        class sovits(Locale.settings.sovits):
            resolve_port_clash_label = "尝试解决端口冲突问题（Windows 可用）"

        class ddsp6(Locale.settings.ddsp6):
            pretrained_model_preference_dropdown_label = "底模偏好"
            default_pretrained_model = "默认底模 512 6"
            large_pretrained_model = "大网络底模 1024 12"

        class ddsp6_1(Locale.settings.ddsp6_1):
            pretrained_model_preference_dropdown_label = "底模偏好"
            default_pretrained_model = "默认(大网络)底模 1024 10"

        saved_tip = "已保存"

    class install_model(Locale.install_model):
        tip = """
        ## 目前仅支持上传 .sf_pkg/.h0_ddsp_pkg_model 格式的模型包
        """

        file_label = "上传模型包"

        model_name_label = "模型名称"
        model_name_placeholder = "请输入模型名称"

        submit_btn_value = "安装模型"

    class path_chooser(Locale.path_chooser):
        input_path_label = "输入文件夹"
        output_path_label = "输出文件夹"

    class fish_audio_preprocess(Locale.fish_audio_preprocess):
        to_wav_tab = "批量转 WAV"
        slice_audio_tab = "切音机"
        preprocess_tab = "数据处理"
        max_duration_label = "最大时长"
        submit_btn_value = "开始"

        input_output_same_tip = "输入输出路径相同"
        input_path_not_exist_tip = "输入路径不存在"

    class vocal_separation(Locale.vocal_separation):
        input_audio_label = "输入音频"
        input_path_label = "输入路径"
        output_path_label = "输出路径"

        use_batch_label = "启用批量处理"
        use_de_reverb_label = "去混响"
        use_harmonic_remove_label = "去和声"

        submit_btn_value = "开始"
        vocal_label = "输出-人声"
        inst_label = "输出-伴奏"

        batch_output_message_label = "批量输出信息"

        no_file_tip = "未选择文件"
        no_input_tip = "未选择输入文件夹"
        no_output_tip = "未选择输出文件夹"
        input_not_exist_tip = "输入文件夹不存在"
        output_not_exist_tip = "输出文件夹不存在"
        input_output_same_tip = "输入输出文件夹相同"

        finished = "完成"
        error_when_processing = "处理时发生错误，可截图控制台寻求帮助"

        unusable_file_tip = "{1} 已跳过, 文件格式不支持"

        batch_progress_desc = "总进度"

        job_to_progress_desc = {
            "vocal": "去人声",
            "kim_vocal": "去人声",
            "deverb": "去混响",
            "karaoke": "去和声",
        }

    class common_infer(Locale.common_infer):
        audio_label = "音频文件"

        use_batch_label = "启用批量处理"

        precision_info = "推理精度"
        precision_label = "精度"

        vocoder_label = "声码器"

        unknown_vocoder_tip = "未知声码器，请检查选项"
        vocoder_not_loaded_tip = "声码器未加载，请检查选项"

        use_vocal_separation_label = "去除伴奏"
        use_vocal_separation_info = "是否去除伴奏"

        use_de_reverb_label = "去除混响"
        use_de_reverb_info = "是否去除混响"

        use_harmonic_remove_label = "去除和声"
        use_harmonic_remove_info = "是否去除和声"

        use_automix_label = "自动混音"
        use_automix_info = "是否使用自动混音（开启去伴奏时生效）"

        automix_but_not_vocal_separation_tip = "自动混音需要开启去除伴奏功能"

        f0_label = "f0 提取器"
        f0_info = "用于音高提取/预测的模型"

        keychange_label = "变调"
        keychange_info = "参考：男转女 12，女转男 -12，音色不像可以调节这个"

        threshold_label = "切片阈值"
        threshold_info = "人声切片的阈值，如果有底噪可以调为 -40 或更高"

        vocal_register_shift_label = "音区偏移"
        vocal_register_shift_info = "音区偏移，可以通过声码器变调让模型唱出更广的音域"
        vocal_register_shift_no_support_tip = (
            "当前声码器不支持音区偏移，音区偏移参数将被忽略"
        )

    class ddsp_based_infer(Locale.ddsp_based_infer):
        method_label = "采样器"
        method_info = "用于 reflow 的采样器"

        infer_step_label = "推理步数"
        infer_step_info = "推理步长，默认就行"

        t_start_label = "T Start"
        t_start_info = "不知道"

        num_formant_shift_key_label = "共振峰偏移"
        num_formant_shift_key_info = "值越大声音越细，值越小声音越粗"

    class ddsp_based_preprocess(Locale.ddsp_based_preprocess):
        method_label = "采样器"
        method_info = "用于 reflow 的采样器"

    class common_preprocess(Locale.common_preprocess):
        encoder_label = "声音编码器"
        encoder_info = "用于对声音进行编码的模型"

        f0_label = "f0 提取器"
        f0_info = "用于音高提取/预测的模型"

    class sovits(Locale.sovits):
        dataset_not_complete_tip = "数据集不完整，请检查数据集或重新预处理"
        finished = "完成"

        class train_main(Locale.sovits.train_main):
            log_interval_label = "日志间隔"
            log_interval_info = "每 N 步输出一次日志"

            eval_interval_label = "验证间隔"
            eval_interval_info = "每 N 步保存一次并验证"

            all_in_mem_label = "缓存全数据集"
            all_in_mem_info = (
                "将所有数据集加载到内存中训练，会加快训练速度，但是需要足够的内存"
            )

            keep_ckpts_label = "保留检查点"
            keep_ckpts_info = "保留最近 N 个检查点"

            batch_size_label = "训练批次大小"
            batch_size_info = "越大越好，越大越占显存"

            learning_rate_label = "学习率"
            learning_rate_info = "学习率"

            num_workers_label = "数据加载器进程数"
            num_workers_info = "仅在 CPU 核心数大于 4 时启用，遵循大就是好原则"

            half_type_label = "精度"
            half_type_info = "选择 fp16 可以获得更快的速度，但是炸炉概率 up up"

        class train_diff(Locale.sovits.train_diff):
            batchsize_label = "训练批次大小"
            batchsize_info = "越大越好，越大越占显存，注意不能超过训练集条数"

            num_workers_label = "训练进程数"
            num_workers_info = "如果你显卡挺好，可以设为 0"

            amp_dtype_label = "训练精度"
            amp_dtype_info = "选择 fp16、bf16 可以获得更快的速度，但是炸炉概率 up up"

            lr_label = "学习率"
            lr_info = "不建议动"

            interval_val_label = "验证间隔"
            interval_val_info = "每 N 步验证一次，同时保存"

            interval_log_label = "日志间隔"
            interval_log_info = "每 N 步输出一次日志"

            interval_force_save_label = "强制保存模型间隔"
            interval_force_save_info = "每 N 步保存一次模型"

            gamma_label = "lr 衰减力度"
            gamma_info = "不建议动"

            cache_device_label = "缓存设备"
            cache_device_info = "选择 cuda 可以获得更快的速度，但是需要更大显存的显卡 (SoVITS 主模型无效)"

            cache_all_data_label = "缓存所有数据"
            cache_all_data_info = "可以获得更快的速度，但是需要大内存/显存的设备"

            epochs_label = "最大训练轮数"
            epochs_info = "达到设定值时将会停止训练"

            use_pretrain_label = "使用预训练模型"
            use_pretrain_info = "勾选可以大幅减少训练时间，如果你不懂，不要动"

        class train_cluster(Locale.sovits.train_cluster):
            cluster_or_index_label = "聚类或检索"
            cluster_or_index_info = "要训练聚类还是检索模型，检索咬字比聚类稍好"

            use_gpu_label = "使用 GPU"
            use_gpu_info = "使用 GPU 可以加速训练，该参数只聚类可用"

        class infer(Locale.sovits.infer):
            cluster_infer_ratio_label = "聚类/特征比例"
            cluster_infer_ratio_info = (
                "聚类/特征占比，范围0-1，若没有训练聚类模型或特征检索则默认0即可"
            )

            linear_gradient_info = "两段音频切片的交叉淡入长度"
            linear_gradient_label = "渐变长度"

            k_step_label = "扩散步数"
            k_step_info = "越大越接近扩散模型的结果，默认100"

            enhancer_adaptive_key_label = "增强器适应"
            enhancer_adaptive_key_info = "使增强器适应更高的音域(单位为半音数)|默认为0"

            f0_filter_threshold_label = "f0 过滤阈值"
            f0_filter_threshold_info = "只有使用crepe时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音"

            audio_predict_f0_label = "自动 f0 预测"
            audio_predict_f0_info = (
                "语音转换自动预测音高，转换歌声时不要打开这个会严重跑调"
            )

            second_encoding_label = "二次编码"
            second_encoding_info = (
                "浅扩散前会对原始音频进行二次编码，玄学选项，有时候效果好，有时候效果差"
            )

            clip_label = "强制切片长度"
            clip_info = "强制音频切片长度, 0 为不强制"

        class preprocess(Locale.sovits.preprocess):
            use_diff_label = "训练浅扩散"
            use_diff_info = "勾选后将会生成训练浅扩散需要的文件，会比不选慢"

            vol_aug_label = "响度嵌入"
            vol_aug_info = "勾选后将会使用响度嵌入"

            num_workers_label = "进程数"
            num_workers_info = "理论越大越快"

            subprocess_num_workers_label = "每个进程的线程数"
            subprocess_num_workers_info = "理论越大越快"

            debug_label = "是否开启 Debug 模式"
            debug_info = "开启后会输出调试信息，非特殊情况没必要开"

        class model_types(Locale.sovits.model_types):
            main = "主模型"
            diff = "浅扩散"
            cluster = "聚类/检索模型"

        class model_chooser_extra(Locale.sovits.model_chooser_extra):
            enhance_label = "NSFHifigan 音频增强"
            enhance_info = (
                "对部分训练集少的模型有一定的音质增强效果，对训练好的模型有反面效果"
            )

            feature_retrieval_label = "启用特征提取"
            feature_retrieval_info = "是否使用特征检索，如果使用聚类模型将被禁用"

            only_diffusion_label = "仅浅扩散"
            only_diffusion_info = "仅推理扩散模型，不推荐"

    class ddsp6(Locale.ddsp6):
        infer_tip = "推理 DDSP 模型"

        class model_types(Locale.ddsp6.model_types):
            cascade = "级联模型"

        class train(Locale.ddsp6.train):
            batch_size_label = "训练批次大小"
            batch_size_info = "越大越好，越大越占显存，注意不能超过训练集条数"

            num_workers_label = "训练进程数"
            num_workers_info = "如果你显卡挺好，可以设为 0"

            amp_dtype_label = "训练精度"
            amp_dtype_info = "选择 fp16、bf16 可以获得更快的速度，但是炸炉概率 up up"

            lr_label = "学习率"
            lr_info = "不建议动"

            interval_val_label = "验证间隔"
            interval_val_info = "每 N 步验证一次，同时保存"

            interval_log_label = "日志间隔"
            interval_log_info = "每 N 步输出一次日志"

            interval_force_save_label = "强制保存模型间隔"
            interval_force_save_info = "每 N 步保存一次模型"

            gamma_label = "lr 衰减力度"
            gamma_info = "不建议动"

            cache_device_label = "缓存设备"
            cache_device_info = "选择 cuda 可以获得更快的速度，但是需要更大显存的显卡 (SoVITS 主模型无效)"

            cache_all_data_label = "缓存所有数据"
            cache_all_data_info = "可以获得更快的速度，但是需要大内存/显存的设备"

            epochs_label = "最大训练轮数"
            epochs_info = "达到设定值时将会停止训练"

            use_pretrain_label = "使用预训练模型"
            use_pretrain_info = "勾选可以大幅减少训练时间，如果你不懂，不要动"

    class reflow(Locale.reflow):
        infer_tip = "推理 ReflowVAESVC 模型"

        class train(Locale.ddsp6.train):
            batch_size_label = "训练批次大小"
            batch_size_info = "越大越好，越大越占显存，注意不能超过训练集条数"

            num_workers_label = "训练进程数"
            num_workers_info = "如果你显卡挺好，可以设为 0"

            amp_dtype_label = "训练精度"
            amp_dtype_info = "选择 fp16、bf16 可以获得更快的速度，但是炸炉概率 up up"

            lr_label = "学习率"
            lr_info = "不建议动"

            interval_val_label = "验证间隔"
            interval_val_info = "每 N 步验证一次，同时保存"

            interval_log_label = "日志间隔"
            interval_log_info = "每 N 步输出一次日志"

            interval_force_save_label = "强制保存模型间隔"
            interval_force_save_info = "每 N 步保存一次模型"

            gamma_label = "lr 衰减力度"
            gamma_info = "不建议动"

            cache_device_label = "缓存设备"
            cache_device_info = "选择 cuda 可以获得更快的速度，但是需要更大显存的显卡 (SoVITS 主模型无效)"

            cache_all_data_label = "缓存所有数据"
            cache_all_data_info = "可以获得更快的速度，但是需要大内存/显存的设备"

            epochs_label = "最大训练轮数"
            epochs_info = "达到设定值时将会停止训练"

            use_pretrain_label = "使用预训练模型"
            use_pretrain_info = "勾选可以大幅减少训练时间，如果你不懂，不要动"

        class model_types(Locale.reflow.model_types):
            cascade = "级联模型"

    default_spk_name = "默认说话人"

    preprocess_draw_desc = "划分验证集"
    preprocess_desc = "预处理(进度去终端看)"
    preprocess_finished = "预处理完成"
