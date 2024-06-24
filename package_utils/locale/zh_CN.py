from .base import Locale


class zhCNLocale(Locale):
    unknown_model_type_tip = "模型类型未知，请手动选择"

    class device_chooser(Locale.device_chooser):
        device_dropdown_label = "设备"

    class model_chooser(Locale.model_chooser):
        submit_btn_value = "选择模型"
        model_type_dropdown_label = "模型类型"
        search_path_label = "搜索路径"

        workdir_name = "工作目录"
        archieve_dir_name = "已归档训练"
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
        textbox_output = "输出结果"

        dorpdown_liked_checkbox_yes = "是"
        dorpdown_liked_checkbox_no = "否"

    class model_manager(Locale.model_manager):
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

    class main_ui(Locale.main_ui):
        release_memory_btn_value = "尝试释放显存/内存"
        released_tip = "已尝试释放显存/内存"
        infer_tab = "💡推理"
        preprocess_tab = "⏳数据处理"
        train_tab = "🏋️‍♂️训练"
        tools_tab = "🛠️小工具"
        settings_tab = "🪡设置"

        model_manager_tab = "模型管理"
        install_model_tab = "安装模型"
        fish_audio_preprocess_tab = "简单音频处理"
        vocal_remove_tab = "人声分离"

        detect_spk_tip = "已检测到的角色："
        spk_not_found_tip = "未检测到任何角色"

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
        little_vram_tip = """
            ## 当前设备没有一张显卡显存大于 6GB，仅推荐训练 DDSP 模型
        """

        open_dataset_folder_btn_value = "打开数据集文件夹"

        choose_model_label = "选择模型"

    class settings(Locale.settings):
        pkg_settings_label = "整合包设置"

        lang_label = "语言"
        lang_info = "更改语言需要重启整合包"

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

        input_path_not_exist_tip = "输入路径不存在"

    class vocal_remove(Locale.vocal_remove):
        input_audio_label = "输入音频"
        submit_btn_value = "开始"
        vocal_label = "输出-人声"
        inst_label = "输出-伴奏"

    class sovits(Locale.sovits):
        dataset_not_complete_tip = "数据集不完整，请检查数据集或重新预处理"
        finished = "完成"

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

        class train(Locale.sovits.train):
            use_diff_label = "训练浅扩散"
            use_diff_info = "勾选后将会生成训练浅扩散需要的文件，会比不选慢"

            vol_aug_label = "响度嵌入"
            vol_aug_info = "勾选后将会使用响度嵌入"

            num_workers_label = "进程数"
            num_workers_info = "理论越大越快"

            subprocess_num_workers_label = "每个进程的线程数"
            subprocess_num_workers_info = "理论越大越快"

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
