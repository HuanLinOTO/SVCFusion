from SVCFusion.locale.base import Locale

locale_name = "en-us"
locale_display_name = "English (US)"


class _Locale(Locale):
    unknown_model_type_tip = "Model type is unknown. Please go to Tools - Model Management to confirm the model type."
    preprocess_failed_tip = "Preprocessing failed! Please take a screenshot of the console information and feedback in the group."
    error_when_infer = "Encountered an error during inference. Skipped {1} files. Check the console for details.{2}"

    class device_chooser(Locale.device_chooser):
        device_dropdown_label = "Device"

    class model_chooser(Locale.model_chooser):
        submit_btn_value = "Choose model"
        model_type_dropdown_label = "Model type."
        search_path_label = "Search Path"
        workdir_name = "Work directory"
        archive_dir_name = "Archived training."
        models_dir_name = "models folder"
        no_model_value = "No model."
        unuse_value = "Do not use."
        no_spk_value = "No speaker"
        choose_model_dropdown_prefix = "Choose model"
        refresh_btn_value = "Refresh Options"
        spk_dropdown_label = "Choose speaker."
        no_spk_option = "No loaded model"

    class form(Locale.form):
        submit_btn_value = "Submit."
        audio_output_1 = "The output result."
        audio_output_2 = "Hello"
        textbox_output = "The output result."
        dorpdown_liked_checkbox_yes = "It is."
        dorpdown_liked_checkbox_no = "No."

    class model_manager(Locale.model_manager):
        choose_model_title = "Choose model"
        action_title = "Action"
        pack_btn_value = "Pack the model."
        pack_result_label = "Packaging result"
        packing_tip = "Hello.  \nPlease do not click multiple times as we are processing your request."
        unpackable_tip = "The model does not support bundling."
        clean_log_btn_value = "Clear logs (confirm no more training before clearing)"
        change_model_type_info = "#### Change Model Type\nUse this only when the model type cannot be identified! Not for converting model types! For changing the recognized model type!"
        change_model_type_btn_value = "Confirm change."
        change_success_tip = "Success in changing."
        change_fail_tip = "Failed to change."
        move_folder_tip = "#### Move to the models directory"
        move_folder_name = "Model name"
        move_folder_name_auto_get = "Automatic retrieval."
        move_folder_btn_value = "Move."
        other_text = "Wait."
        moving_tip = "Moving, please do not click multiple times."
        moved_tip = "Moved to {1}, can be used after refreshing."

    class main_ui(Locale.main_ui):
        release_memory_btn_value = "Try releasing GPU/RAM memory."
        released_tip = "Attempting to release GPU/graphics memory."
        infer_tab = "💡 Deduction"
        preprocess_tab = "Processing data."
        train_tab = "Working out."
        tools_tab = "Toolbox."
        settings_tab = "Twist setting."
        model_tools_tab = "Model-related"
        audio_tools_tab = "Audio related."
        realtime_tools_tab = "Real-time"
        start_ddsp_realtime_gui_btn = "Start the DDSP Real-Time GUI"
        starting_tip = (
            '"Starting up. Please wait. Do not click again, consequences are severe."'
        )
        load_model_btn_value = "Load model."
        infer_btn_value = "Start reasoning."
        model_manager_tab = "Model Management"
        install_model_tab = "Install model"
        fish_audio_preprocess_tab = "Simple audio processing."
        vocal_separation_tab = "Output: Speech Separation"
        compatible_tab = "Model compatibility"
        detect_spk_tip = "Detected roles:"
        spk_not_found_tip = "No roles detected."

    class compatible_models(Locale.compatible_models):
        upload_error = "Upload error. Please check if the file is complete."
        model_name_label = "Model name"
        upload_success = "Upload successful."
        model_exists = "The model already exists."
        compatible_sovits = "SOVITS model compatibility"
        sovits_main_model_label = "SoVITS main model"
        sovits_diff_model_label = "SoVITS Shallow diffusion"
        sovits_cluster_model_label = "SoVITS clustering/searching"
        sovits_main_model_config_label = "SoVITS main model configuration"
        sovits_diff_model_config_label = "SoVITS shallow diffusion configuration"

    class preprocess(Locale.preprocess):
        tip = "Please first place your dataset, which is a bunch of `.wav` files, into the `dataset_raw/[YourCharacterName]` folder under the integration package.\n\nYou can train multiple characters simultaneously by creating separate folders for each character.\n\nAfter placing them, your directory should look like this:\n\n```\ndataset_raw/\n|-[YourCharacterName1]/\n  |-1.wav\n  |-2.wav\n  |-3.wav\n  ...\n|-[YourCharacterName2]/\n  |-1.wav\n  |-2.wav\n  |-3.wav\n  ...\n```\n\nIf you don't understand anything, simply click the button below for automatic data processing.\n\nIf you are familiar with the meaning of parameters, switch to manual mode for more detailed data processing.\n\n**For CPU users, please use FCPE as the F0 extractor/predictor.**"
        low_vram_tip = "## Current device does not have a GPU with more than 6GB of memory, only the training of the DDSP model is recommended.\n\nNote This does not mean you cannot train!!"
        open_dataset_folder_btn_value = "Open the dataset folder."
        choose_model_label = "Choose model"
        start_preprocess_btn_value = "Start preprocessing."

    class train(Locale.train):
        current_train_model_label = "Current training model"
        fouzu_tip = '"Made a cyberspace Buddha for you. I hope it helps."'
        gd_plus_1 = "Click me to add merit."
        gd_plus_1_tip = "Good karma +1, fryer -1"
        choose_sub_model_label = "Choose sub-model"
        start_train_btn_value = "Start/Continue Training"
        archive_btn_value = "Archived work directory"
        stop_btn_value = "Stop training."
        archieving_tip = "Archiving... Do not click multiple times."
        archived_tip = "Archive complete, please check the opened folder."
        stopped_tip = "Hello.\nOutput: The stop training command has been sent. Please check the training window."
        tensorboard_btn = "Start Tensorboard"
        launching_tb_tip = 'Hello.\nOutput: \n"Starting Tensorboard, please wait."'
        launched_tb_tip = "Tensorboard has been opened at {1}."

    class settings(Locale.settings):
        page = "Page"
        save_btn_value = "Save settings"
        pkg_settings_label = "Integrate package settings"
        infer_settings_label = "Inference settings"
        sovits_settings_label = "Setting up So-VITS-SVC."
        ddsp6_settings_label = "DDSP-SVC settings configuration."

        class pkg(Locale.settings.pkg):
            lang_label = "Language"
            lang_info = (
                "Language needs to be restarted for the integration package to change."
            )

        class infer(Locale.settings.infer):
            msst_device_label = "Run separation task using device."

        class sovits(Locale.settings.sovits):
            resolve_port_clash_label = (
                "Try resolving port conflict issues (Windows is applicable)"
            )

        class ddsp6(Locale.settings.ddsp6):
            pretrained_model_preference_dropdown_label = (
                "Preference for baseline models"
            )
            default_pretrained_model = "Default base model is 512x6."
            large_pretrained_model = "Large network base model 1024 x 12"

        class ddsp6_1(Locale.settings.ddsp6_1):
            pretrained_model_preference_dropdown_label = (
                "Preference for baseline models"
            )
            default_pretrained_model = "Default base model is 512x6."
            large_pretrained_model = "Large network base model 1024 x 12"

        saved_tip = "Saved."

    class install_model(Locale.install_model):
        tip = "## Currently only supports uploading model packages in .sf_pkg/.h0_ddsp_pkg_model formats."
        file_label = "Upload model package"
        model_name_label = "Model name"
        model_name_placeholder = "Please enter the model name."
        submit_btn_value = "Install model"

    class path_chooser(Locale.path_chooser):
        input_path_label = "Input: 输出文件夹"
        output_path_label = "Output folder"

    class fish_audio_preprocess(Locale.fish_audio_preprocess):
        to_wav_tab = "Batch convert to WAV"
        slice_audio_tab = "Cephalopod"
        preprocess_tab = "Data processing"
        max_duration_label = "Longest duration"
        submit_btn_value = "Start."
        input_output_same_tip = (
            "Input: 输入输出路径相同\nOutput: The input and output paths are the same."
        )
        input_path_not_exist_tip = "The input path does not exist."

    class vocal_separation(Locale.vocal_separation):
        input_audio_label = "Input: 输出音频"
        input_path_label = "Input: 输出文件\nOutput: Export file"
        output_path_label = "Path for output"
        use_batch_label = "Enable bulk processing"
        use_de_reverb_label = "Go to reverb."
        use_harmonic_remove_label = "Chord out."
        submit_btn_value = "Start."
        vocal_label = "Hello."
        inst_label = "Hello."
        batch_output_message_label = "Batch output information"
        no_file_tip = "No file selected."
        no_input_tip = "No selected folder."
        no_output_tip = "No output folder selected."
        input_not_exist_tip = "The input folder does not exist."
        output_not_exist_tip = "The output folder does not exist."
        input_output_same_tip = "The input folder and output folder are the same."
        finished = "Done."
        error_when_processing = "An error occurred during processing. You can seek help by taking a screenshot of the console."
        unusable_file_tip = "{1} has been skipped. The file format is not supported."
        batch_progress_desc = "Total progress"
        job_to_progress_desc = {
            "Vocal.": "Silence.",
            "kim_vocal": "Silence.",
            "There seems to be a mistake in your request. If you need translation from Chinese to English or any other kind of textual translation service and also want it kept as Markdown format, please provide the text you want translated along with its specific usage context if possible. \n\nFor instance:\n\nInput: 你好。\nResult: Hello.\n\nIn case there are multiple inputs:\n\nInput: 很高兴见到你！欢迎来到我的世界。\nResult: Nice to meet you! Welcome to my world.": "Go to reverb.",
            "Karaoke.": "Chord out.",
        }

    class common_infer(Locale.common_infer):
        audio_label = "Audio file"
        use_batch_label = "Enable bulk processing"
        use_vocal_separation_label = "Remove accompaniment."
        use_vocal_separation_info = "Should we remove the backing track?"
        use_de_reverb_label = "Remove reverb."
        use_de_reverb_info = "Do you want to remove reverb?"
        use_harmonic_remove_label = "Remove harmony."
        use_harmonic_remove_info = "Do you want to remove the harmonics?"
        f0_label = "f0 Extractor"
        f0_info = "Model for pitch extraction/prediction."
        keychange_label = "Tune change"
        keychange_info = "Reference: Male-to-female is 12, female-to-male is -12. The sound quality can be adjusted in this way if it doesn't sound right."
        threshold_label = "Slice threshold"
        threshold_info = "Threshold for voice clip segmentation. If there is background noise, you can adjust it to -40 or higher."

    class ddsp_based_infer(Locale.ddsp_based_infer):
        method_label = "Sampler"
        method_info = "Sampler for reflow"
        infer_step_label = "Hello."
        infer_step_info = "Inference step length, default is fine."
        t_start_label = "T Start"
        t_start_info = "I don't know."
        num_formant_shift_key_label = "Resonance peak shift"
        num_formant_shift_key_info = "The higher the value, the finer the sound; the lower the value, the rougher the sound."

    class ddsp_based_preprocess(Locale.ddsp_based_preprocess):
        method_label = "f0 Extractor"
        method_info = "Sampler for reflow"

    class common_preprocess(Locale.common_preprocess):
        encoder_label = "Audio encoder"
        encoder_info = "Model used for encoding sound."
        f0_label = "f0 Extractor"
        f0_info = "Model for pitch extraction/prediction."

    class sovits(Locale.sovits):
        dataset_not_complete_tip = (
            "The dataset is incomplete, please check the dataset or re-preprocess."
        )
        finished = "Done."

        class train_main(Locale.sovits.train_main):
            log_interval_label = "Log interval"
            log_interval_info = "Log once every N steps."
            eval_interval_label = "Verification interval"
            eval_interval_info = "Save and verify every N steps."
            all_in_mem_label = "Cache full dataset"
            all_in_mem_info = "To train with all datasets loaded into memory can speed up the training process, but it requires sufficient memory."
            keep_ckpts_label = "Keep checkpoints."
            keep_ckpts_info = "Keep the last N checkpoints."
            batch_size_label = "Training batch size"
            batch_size_info = (
                "Larger is better, the larger the more it occupies GPU memory."
            )
            learning_rate_label = "Learning rate"
            learning_rate_info = "Learning rate"
            num_workers_label = "Number of processes for data loader."
            num_workers_info = "Enable only if the number of CPU cores is greater than 4, adhering to the principle that bigger is better."
            half_type_label = "Accuracy"
            half_type_info = "Choosing fp16 can give you faster speeds, but it increases the risk of overloading your system."

        class train_diff(Locale.sovits.train_diff):
            batchsize_label = "Training batch size"
            batchsize_info = "Larger is better but it also consumes more GPU memory. Be sure not to exceed the number of samples in your training set."
            num_workers_label = "Number of training processes"
            num_workers_info = "If your graphics card is good, you can set it to 0."
            amp_dtype_label = "Training accuracy"
            amp_dtype_info = "Choosing fp16 or bf16 can give you faster speeds, but the risk of hardware failure is significantly increased."
            lr_label = "Learning rate"
            lr_info = "It is not recommended to move."
            interval_val_label = "Verification interval"
            interval_val_info = "Check every N steps and save at the same time."
            interval_log_label = "Log interval"
            interval_log_info = "Log once every N steps."
            interval_force_save_label = "Force save model interval"
            interval_force_save_info = "Save the model every N steps."
            gamma_label = "Learning rate decay strength"
            gamma_info = "It is not recommended to move."
            cache_device_label = "Cache device"
            cache_device_info = "Choosing CUDA can provide faster speeds, but requires a graphics card with more VRAM (The SoVITS main model is invalid)"
            cache_all_data_label = "Cache all data."
            cache_all_data_info = "You can achieve faster speeds, but it requires devices with large memory or graphics memory."
            epochs_label = "Max training epochs"
            epochs_info = "Training will stop when reaching the set value."
            use_pretrain_label = "Use a pre-trained model"
            use_pretrain_info = "Selecting can significantly reduce training time. If you don't understand, don't touch it."

        class train_cluster(Locale.sovits.train_cluster):
            cluster_or_index_label = "Clustering or retrieval"
            cluster_or_index_info = "To train a clustering or retrieval model, retrieval performs slightly better than clustering."
            use_gpu_label = "Use GPU"
            use_gpu_info = "Using a GPU can accelerate training. This parameter only clusters available ones."

        class infer(Locale.sovits.infer):
            cluster_infer_ratio_label = "Clustering/Feature Ratio"
            cluster_infer_ratio_info = "Clustering/feature proportionality. The range is 0 to 1. If no clustering model or feature retrieval has been trained, it defaults to 0."
            linear_gradient_info = "Crossfade length between two audio slice overlaps."
            linear_gradient_label = "Gradient length"
            k_step_label = "Expand steps."
            k_step_info = "The larger the value, the closer it gets to the result of the diffusion model, default is 100."
            enhancer_adaptive_key_label = "Enhancer adaptation"
            enhancer_adaptive_key_info = '"Adjust enhancer to a higher pitch range (in semitones) | Default is 0"'
            f0_filter_threshold_label = "f0 filtering threshold"
            f0_filter_threshold_info = "Only effective when using crepe. The numeric range is from 0 to 1. Decreasing this value reduces the pitch deviation probability but increases the number of dead notes."
            audio_predict_f0_label = "Automatic f0 prediction."
            audio_predict_f0_info = "Voice conversion automatically predicts pitch. Don't turn on this when converting singing voice as it can seriously off-tune the melody."
            second_encoding_label = "Second encoding"
            second_encoding_info = "Before shallow diffusion, the original audio will undergo secondary encoding. It's a mystical option; sometimes it yields good results, other times it doesn't work as well."
            clip_label = "Forced slice length"
            clip_info = "Force audio slice length. 0 means no forced slicing."

        class preprocess(Locale.sovits.preprocess):
            use_diff_label = "Train shallow diffusion."
            use_diff_info = "Checking this will result in the generation of files necessary for training shallow diffusion. It will be slower compared to not checking it."
            vol_aug_label = "Output: Loudness embedding"
            vol_aug_info = "If checked, this will use loudness embedding."
            num_workers_label = "Number of processes."
            num_workers_info = "The bigger the theory, the faster it goes."
            subprocess_num_workers_label = "The number of threads for each process."
            subprocess_num_workers_info = "The bigger the theory, the faster it goes."
            debug_label = "Is Debug mode enabled?"
            debug_info = "Opening it will output debug information. There's no need to open in most non-special cases."

        class model_types(Locale.sovits.model_types):
            main = "Master model"
            diff = "Shallow diffusion"
            cluster = "Clustering/retrieval model"

        class model_chooser_extra(Locale.sovits.model_chooser_extra):
            enhance_label = "Audio Enhancement for NSFHI-FIGAN"
            enhance_info = "The model has a certain sound enhancement effect for datasets with less training data and a negative effect on well-trained models."
            feature_retrieval_label = "Enable feature extraction"
            feature_retrieval_info = "Question: Is feature retrieval used? If so, clustering models will be disabled."

    class ddsp6(Locale.ddsp6):
        infer_tip = "Inferential DDSP Model"

        class model_types(Locale.ddsp6.model_types):
            cascade = "Cascaded model"

        class train(Locale.ddsp6.train):
            batch_size_label = "Training batch size"
            batch_size_info = "Larger is better but it also consumes more GPU memory. Be sure not to exceed the number of samples in your training set."
            num_workers_label = "Number of training processes"
            num_workers_info = "If your graphics card is good, you can set it to 0."
            amp_dtype_label = "Training accuracy"
            amp_dtype_info = "Choosing fp16 or bf16 can give you faster speeds, but the risk of hardware failure is significantly increased."
            lr_label = "Learning rate"
            lr_info = "It is not recommended to move."
            interval_val_label = "Verification interval"
            interval_val_info = "Check every N steps and save at the same time."
            interval_log_label = "Log interval"
            interval_log_info = "Log once every N steps."
            interval_force_save_label = "Force save model interval"
            interval_force_save_info = "Save the model every N steps."
            gamma_label = "Learning rate decay strength"
            gamma_info = "It is not recommended to move."
            cache_device_label = "Cache device"
            cache_device_info = "Choosing CUDA can provide faster speeds, but requires a graphics card with more VRAM (The SoVITS main model is invalid)"
            cache_all_data_label = "Cache all data."
            cache_all_data_info = "You can achieve faster speeds, but it requires devices with large memory or graphics memory."
            epochs_label = "Max training epochs"
            epochs_info = "Training will stop when reaching the set value."
            use_pretrain_label = "Use a pre-trained model"
            use_pretrain_info = "Selecting can significantly reduce training time. If you don't understand, don't touch it."

    class reflow(Locale.reflow):
        infer_tip = "推理 Re-flow VAE SV-C model"

        class train(Locale.ddsp6.train):
            batch_size_label = "Training batch size"
            batch_size_info = "Larger is better but it also consumes more GPU memory. Be sure not to exceed the number of samples in your training set."
            num_workers_label = "Number of training processes"
            num_workers_info = "If your graphics card is good, you can set it to 0."
            amp_dtype_label = "Training accuracy"
            amp_dtype_info = "Choosing fp16 or bf16 can give you faster speeds, but the risk of hardware failure is significantly increased."
            lr_label = "Learning rate"
            lr_info = "It is not recommended to move."
            interval_val_label = "Verification interval"
            interval_val_info = "Check every N steps and save at the same time."
            interval_log_label = "Log interval"
            interval_log_info = "Log once every N steps."
            interval_force_save_label = "Force save model interval"
            interval_force_save_info = "Save the model every N steps."
            gamma_label = "Learning rate decay strength"
            gamma_info = "It is not recommended to move."
            cache_device_label = "Cache device"
            cache_device_info = "Choosing CUDA can provide faster speeds, but requires a graphics card with more VRAM (The SoVITS main model is invalid)"
            cache_all_data_label = "Cache all data."
            cache_all_data_info = "You can achieve faster speeds, but it requires devices with large memory or graphics memory."
            epochs_label = "Max training epochs"
            epochs_info = "Training will stop when reaching the set value."
            use_pretrain_label = "Use a pre-trained model"
            use_pretrain_info = "Selecting can significantly reduce training time. If you don't understand, don't touch it."

        class model_types(Locale.reflow.model_types):
            cascade = "Cascaded model"

    default_spk_name = "Default speaker"
    preprocess_draw_desc = "Split validation set."
    preprocess_desc = "Preprocessing (check progress in the terminal)."
    preprocess_finished = "Preprocessing is complete."
