
from package_utils.locale.base import Locale

locale_name = "en-us"
locale_display_name = "English (US)"

class _Locale(Locale):
    unknown_model_type_tip = "Unknown model type, please select manually"
    class device_chooser(Locale.device_chooser):
        device_dropdown_label = "Device"
    
    class model_chooser(Locale.model_chooser):
        submit_btn_value = "Select Model"
        model_type_dropdown_label = "Model Type"
        search_path_label = "Search Path"
        workdir_name = "Working Directory"
        archieve_dir_name = "Archived Training"
        models_dir_name = "Models Folder"
        no_model_value = "No Model"
        unuse_value = "Do Not Use"
        no_spk_value = "No Speaker"
        choose_model_dropdown_prefix = "Choose Model"
        refresh_btn_value = "Refresh Options"
        spk_dropdown_label = "Select Speaker"
        no_spk_option = "No Model Loaded"
    
    class form(Locale.form):
        submit_btn_value = "Submit"
        audio_output_1 = "Output Result"
        audio_output_2 = "Output Result/Accompaniment"
        textbox_output = "Output Result"
        dorpdown_liked_checkbox_yes = "Yes"
        dorpdown_liked_checkbox_no = "No"
    
    class model_manager(Locale.model_manager):
        pack_btn_value = "Pack Model"
        pack_result_label = "Pack Result"
        packing_tip = "Packing, please do not click multiple times"
        unpackable_tip = "This model does not support packing"
        clean_log_btn_value = "Clear Log (Clear only if no longer training)"
        change_model_type_info = """
    
        
            
                
                    
                            #### Change Model Type
                            Use only if the model type cannot be recognized! This is not for converting the model type! It's for changing the recognized model type!
                            
                    
                
            
        
    """
        change_model_type_btn_value = "Confirm Change"
        change_success_tip = "Change Successful"
        change_fail_tip = "Change Failed"
    
    class main_ui(Locale.main_ui):
        release_memory_btn_value = "Try to Release VRAM/Memory"
        released_tip = "Attempted to release VRAM/Memory"
        infer_tab = "💡Inference"
        preprocess_tab = "⏳Data Processing"
        train_tab = "🏋️‍♂️Training"
        tools_tab = "🛠️Tools"
        settings_tab = "🪡Settings"
        model_manager_tab = "Model Management"
        install_model_tab = "Install Model"
        fish_audio_preprocess_tab = "Simple Audio Processing"
        vocal_remove_tab = "Vocal Removal"
        detect_spk_tip = "Detected Characters:"
        spk_not_found_tip = "No Characters Detected"
    
    class preprocess(Locale.preprocess):
        tip = """
    
        
            
                
                    
                                Please put your dataset (a bunch of `.wav` files) into the `dataset_raw/Your Character Name` folder under the integration package
                    
                                You can train multiple characters at the same time by creating multiple character folders
                    
                                Once placed, your directory should look like this:
                    
                                ```
                                dataset_raw/
                                |-Your Character Name1/
                                |  | 1.wav
                                |  | 2.wav
                                |  | 3.wav
                                |  ...
                                |-Your Character Name2/
                                |  | 1.wav
                                |  | 2.wav
                                |  | 3.wav
                                |  ...
                                ```
                    
                                If you don't understand anything, just click the button below for automatic data processing
                    
                                If you understand the parameters, you can switch to manual mode for more detailed data processing
                                
                                **CPU users, please use FCPE as the F0 extractor/predictor**
                            
                    
                
            
        
    """
        little_vram_tip = """
    
        
            
                
                    
                                ## The current device does not have a GPU with more than 6GB of VRAM, only training DDSP models is recommended
                            
                    
                
            
        
    """
        open_dataset_folder_btn_value = "Open Dataset Folder"
        choose_model_label = "Choose Model"
    
    class train(Locale.train):
        current_train_model_label = "Current Training Model"
        fouzu_tip = "This is the Buddha from Buddhist culture who can bless your AI training smoothly."
        gd_plus_1 = "Click to Add Merit"
        gd_plus_1_tip = "Merit +1, Furnace Explosion -1"
        choose_sub_model_label = "Choose Sub Model"
        archieve_btn_value = "Archive Working Directory"
        stop_btn_value = "Stop Training"
        archieving_tip = "Archiving, please do not click multiple times"
        archieved_tip = "Archiving Complete, please check the opened folder"
        stopped_tip = "Stop training command sent, please check the training window"
        tensorboard_btn = "启动 Tensorboard"
        launching_tb_tip = "正在启动 Tensorboard，请稍后"
        launched_tb_tip = "Tensorboard 已在 {1} 开放"
    
    class settings(Locale.settings):
        page = "Page"
        pkg_settings_label = "Integration Package Settings"
        sovits_settings_label = "SoVITS Settings"
        class pkg(Locale.settings.pkg):
            lang_label = "Language"
            lang_info = "Changing the language requires restarting the integration package"
        
        class sovits(Locale.settings.sovits):
            resolve_port_clash_label = "Attempt to Resolve Port Conflict Issues"
        
        saved_tip = "Saved"
        ddsp6_settings_label = "DDSP-SVC 6 设置"
        class ddsp6(Locale.settings.ddsp6):
            pretrained_model_preference_dropdown_label = "底模偏好"
            default_pretrained_model = "默认底模 512 x 6"
            large_pretrained_model = "大网络底模 1024 x 6"
        
    
    class install_model(Locale.install_model):
        tip = """
    
        
            
                
                    
                            ## Currently only supports uploading .sf_pkg/.h0_ddsp_pkg_model format model packages
                            
                    
                
            
        
    """
        file_label = "Upload Model Package"
        model_name_label = "Model Name"
        model_name_placeholder = "Enter Model Name"
        submit_btn_value = "Install Model"
    
    class path_chooser(Locale.path_chooser):
        input_path_label = "Input Folder"
        output_path_label = "Output Folder"
    
    class fish_audio_preprocess(Locale.fish_audio_preprocess):
        to_wav_tab = "Batch to WAV"
        slice_audio_tab = "Audio Slicing"
        preprocess_tab = "Data Processing"
        max_duration_label = "Max Duration"
        submit_btn_value = "Start"
        input_path_not_exist_tip = "Input path does not exist"
    
    class vocal_remove(Locale.vocal_remove):
        input_audio_label = "Input Audio"
        submit_btn_value = "Start"
        vocal_label = "Output - Vocal"
        inst_label = "Output - Accompaniment"
    
    class common_infer(Locale.common_infer):
        audio_label = "Audio File"
        use_vocal_remove_label = "Remove Accompaniment"
        use_vocal_remove_info = "Whether to remove accompaniment"
        f0_label = "F0 Extractor"
        f0_info = "Model for pitch extraction/prediction"
        keychange_label = "Pitch Shift"
        keychange_info = "Reference: male to female 12, female to male -12, adjust this if the timbre is not right"
        threshold_label = "Slicing Threshold"
        threshold_info = "Slicing threshold for vocals, adjust to -40 or higher if there is background noise"
        use_harmony_remove_label = "去除和声"
        use_harmony_remove_info = "是否去除和声（必须先勾选移除伴奏）"
    
    class diff_based_infer(Locale.diff_based_infer):
        method_label = "Sampler"
        method_info = "Sampler for reflow"
        infer_step_label = "Inference Steps"
        infer_step_info = "Inference steps, default is fine"
        t_start_label = "T Start"
        t_start_info = "Unknown"
    
    class diff_based_preprocess(Locale.diff_based_preprocess):
        method_label = "F0 Extractor"
        method_info = "Sampler for reflow"
    
    class common_preprocess(Locale.common_preprocess):
        encoder_label = "Voice Encoder"
        encoder_info = "Model for encoding voice"
        f0_label = "F0 Extractor"
        f0_info = "Model for pitch extraction/prediction"
    
    class sovits(Locale.sovits):
        dataset_not_complete_tip = "Dataset incomplete, please check the dataset or reprocess"
        finished = "Finished"
        class infer(Locale.sovits.infer):
            cluster_infer_ratio_label = "Cluster/Feature Ratio"
            cluster_infer_ratio_info = "Ratio of clustering/features, range 0-1, if no clustering model or feature retrieval is trained, default is 0"
            linear_gradient_info = "Crossfade length between two audio slices"
            linear_gradient_label = "Fade Length"
            k_step_label = "Diffusion Steps"
            k_step_info = "The larger the number, the closer to the result of the diffusion model, default is 100"
            enhancer_adaptive_key_label = "Enhancer Adaptive"
            enhancer_adaptive_key_info = "Make the enhancer adapt to higher pitches (unit is semitones), default is 0"
            f0_filter_threshold_label = "F0 Filter Threshold"
            f0_filter_threshold_info = "Effective only when using crepe. Range from 0-1. Lowering this value reduces the probability of pitch deviation but increases mute sounds"
            audio_predict_f0_label = "Automatic F0 Prediction"
            audio_predict_f0_info = "Automatic pitch prediction for voice conversion, do not enable this when converting singing voices as it will cause serious pitch deviations"
            second_encoding_label = "Secondary Encoding"
            second_encoding_info = "The original audio will be re-encoded before shallow diffusion, an occult option that sometimes works well and sometimes poorly"
            clip_label = "Force Slice Length"
            clip_info = "Force audio slice length, 0 means no force"
        
        class preprocess(Locale.sovits.preprocess):
            use_diff_label = "Train Shallow Diffusion"
            use_diff_info = "Check this to generate files needed for training shallow diffusion, will be slower than not checking"
            vol_aug_label = "Loudness Embedding"
            vol_aug_info = "Check this to use loudness embedding"
            num_workers_label = "Number of Processes"
            num_workers_info = "Theoretically the larger the faster"
            subprocess_num_workers_label = "Threads per Process"
            subprocess_num_workers_info = "Theoretically the larger the faster"
            debug_label = "是否开启 Debug 模式"
            debug_info = "开启后会输出调试信息，非特殊情况没必要开"
        
        class model_types(Locale.sovits.model_types):
            main = "Main Model"
            diff = "Shallow Diffusion"
            cluster = "Clustering/Retrieval Model"
        
        class model_chooser_extra(Locale.sovits.model_chooser_extra):
            enhance_label = "NSFHifigan Audio Enhancement"
            enhance_info = "Provides certain audio quality enhancement for some models with less training data, but has a negative effect on well-trained models"
            feature_retrieval_label = "Enable Feature Extraction"
            feature_retrieval_info = "Whether to use feature retrieval, will be disabled if using clustering model"
        
        class train_main(Locale.sovits.train_main):
            log_interval_label = "日志间隔"
            log_interval_info = "每 N 步输出一次日志"
            eval_interval_label = "验证间隔"
            eval_interval_info = "每 N 步保存一次并验证"
            all_in_mem_label = "缓存全数据集"
            all_in_mem_info = "将所有数据集加载到内存中训练，会加快训练速度，但是需要足够的内存"
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
        
    
    class ddsp6(Locale.ddsp6):
        infer_tip = "Inference DDSP Model"
        class model_types(Locale.ddsp6.model_types):
            cascade = "Cascade Model"
        
        class train(Locale.ddsp6.train):
            batch_size_label = "Training Batch Size"
            batch_size_info = "The larger the better, the larger the more VRAM required, ensure it does not exceed the number of training sets"
            num_workers_label = "Training Processes"
            num_workers_info = "If your GPU is good, set to 0"
            amp_dtype_label = "Training Precision"
            amp_dtype_info = "Selecting fp16 or bf16 can achieve faster speed but increases the probability of crashes"
            lr_label = "Learning Rate"
            lr_info = "Not recommended to change"
            interval_val_label = "Validation Interval"
            interval_val_info = "Validate every N steps, and save"
            interval_log_label = "Log Interval"
            interval_log_info = "Output log every N steps"
            interval_force_save_label = "Force Save Model Interval"
            interval_force_save_info = "Save model every N steps"
            gamma_label = "LR Decay Strength"
            gamma_info = "Not recommended to change"
            cache_device_label = "Cache Device"
            cache_device_info = "Selecting cuda can achieve faster speed but requires larger VRAM (Invalid for SoVITS main model)"
            cache_all_data_label = "Cache All Data"
            cache_all_data_info = "Can achieve faster speed but requires devices with large memory/VRAM"
            epochs_label = "Max Training Epochs"
            epochs_info = "Will stop training when reaching the set value"
            use_pretrain_label = "Use Pretrained Model"
            use_pretrain_info = "Checking this can greatly reduce training time, if you don't understand, don't touch"
        
    
    class reflow(Locale.reflow):
        infer_tip = "Inference ReflowVAESVC Model"
        class train(Locale.reflow.train):
            batch_size_label = "Training Batch Size"
            batch_size_info = "The larger the better, the larger the more VRAM required, ensure it does not exceed the number of training sets"
            num_workers_label = "Training Processes"
            num_workers_info = "If your GPU is good, set to 0"
            amp_dtype_label = "Training Precision"
            amp_dtype_info = "Selecting fp16 or bf16 can achieve faster speed but increases the probability of crashes"
            lr_label = "Learning Rate"
            lr_info = "Not recommended to change"
            interval_val_label = "Validation Interval"
            interval_val_info = "Validate every N steps, and save"
            interval_log_label = "Log Interval"
            interval_log_info = "Output log every N steps"
            interval_force_save_label = "Force Save Model Interval"
            interval_force_save_info = "Save model every N steps"
            gamma_label = "LR Decay Strength"
            gamma_info = "Not recommended to change"
            cache_device_label = "Cache Device"
            cache_device_info = "Selecting cuda can achieve faster speed but requires larger VRAM (Invalid for SoVITS main model)"
            cache_all_data_label = "Cache All Data"
            cache_all_data_info = "Can achieve faster speed but requires devices with large memory/VRAM"
            epochs_label = "Max Training Epochs"
            epochs_info = "Will stop training when reaching the set value"
            use_pretrain_label = "Use Pretrained Model"
            use_pretrain_info = "Checking this can greatly reduce training time, if you don't understand, don't touch"
        
        class model_types(Locale.reflow.model_types):
            cascade = "Cascade Model"
        
    
    default_spk_name = "Default Speaker"
    preprocess_draw_desc = "Divide Validation Set"
    preprocess_desc = "Preprocess (see progress in terminal)"
    preprocess_finished = "Preprocessing Complete"
