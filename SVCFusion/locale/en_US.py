
from SVCFusion.locale.base import Locale

locale_name = "en-us"
locale_display_name = "English (US)"

class _Locale(Locale): 
    unknown_model_type_tip = "Unknown model type, please go to Tools-Model Management to confirm the model type."
    preprocess_failed_tip = "Preprocessing failed! Please take a screenshot of the console information and join the group for feedback."

    error_when_infer = "Error occurred during inference<br>Skipped {1} file(s)<br>Check console for details<br>{2}"

    class device_chooser(Locale.device_chooser):
        device_dropdown_label = "Device"

    class model_chooser(Locale.model_chooser):
        submit_btn_value = "Select Model"
        model_type_dropdown_label = "Model Type"
        search_path_label = "Search Path"

        workdir_name = "Working Directory"
        archive_dir_name = "Archived Training"
        models_dir_name = "models Folder"

        no_model_value = "No Model"
        unuse_value = "Do Not Use"
        no_spk_value = "No Speaker"

        choose_model_dropdown_prefix = "Choose Model"

        refresh_btn_value = "Refresh Options"

        spk_dropdown_label = "Select Speaker"
        no_spk_option = "Model not loaded"

    class form(Locale.form):
        submit_btn_value = "Submit"
        audio_output_1 = "Output Result"
        audio_output_2 = "Output Result/Track"
        textbox_output = "Output Result"

        dorpdown_liked_checkbox_yes = "Yes"
        dorpdown_liked_checkbox_no = "No"

    class model_manager(Locale.model_manager):
        choose_model_title = "Choose Model"
        action_title = "Action"

        pack_btn_value = "Pack Model"
        pack_result_label = "Packing Result"
        packing_tip = "Packing in progress, do not click repeatedly."
        unpackable_tip = "This model does not support packaging."

        clean_log_btn_value = "Clear Logs (Confirm no further training before clearing)"

        change_model_type_info = """
        #### Change Model Type
        Use only if the model type cannot be recognized! Not for converting model types, but changing the recognized model type!
        """
        change_model_type_btn_value = "Confirm Change"
        change_success_tip = "Change successful"
        change_fail_tip = "Change failed"

        move_folder_tip = "#### Move to models Directory"
        move_folder_name = "Model Name"
        move_folder_name_auto_get = "Auto Get"
        move_folder_btn_value = "Move"
        other_text = "etc."
        moving_tip = "Moving in progress, do not click repeatedly."
        moved_tip = "Moved to {1}, refresh available."

    class main_ui(Locale.main_ui):
        release_memory_btn_value = "Try to Free GPU/Memory"
        released_tip = "Tried to free GPU/Memory"
        infer_tab = "💡Inference"
        preprocess_tab = "⏳Data Processing"
        train_tab = "🏋️‍♂️Training"
        tools_tab = "🛠️Tools"
        settings_tab = "⚙️Settings"

        model_tools_tab = "Model Related"
        audio_tools_tab = "Audio Related"
        realtime_tools_tab = "Realtime"
        dlc_install_tools_tab = "DLC Install"

        start_ddsp_realtime_gui_btn = "Start DDSP Real-time GUI"

        starting_tip = "Starting, please wait. Do not click repeatedly."

        load_model_btn_value = "Load Model"
        infer_btn_value = "Start Inference"

        model_manager_tab = "Model Management"
        install_model_tab = "Install Model"
        fish_audio_preprocess_tab = "Simple Audio Preprocessing"
        vocal_separation_tab = "Vocal Separation"
        compatible_tab = "Compatibility"

        detect_spk_tip = "Detected characters:"
        spk_not_found_tip = "No characters detected"

    class DLC(Locale.DLC):
        dlc_install_label = "Upload New DLC"
        dlc_install_btn_value = "Install DLC"
        dlc_installing_tip = "Installing"
        dlc_install_success = "Installation Successful"
        dlc_install_failed = "Installation Failed"
        dlc_install_empty = "No file selected"
        dlc_install_ext_error = "Unsupported .sf_dlc file format"

    class compatible_models(Locale.compatible_models):
        upload_error = "Upload Error, please check if the file is complete."
        model_name_label = "Model Name"
        upload_success = "Upload Successful"
        model_exists = "Model Already Exists"

        compatible_sovits = "SoVITS Model Compatibility"
        sovits_main_model_label = "SoVITS Main Model"
        sovits_diff_model_label = "SoVITS Diffusion"
        sovits_cluster_model_label = "SoVITS Clustering/Retrieval"

        sovits_main_model_config_label = "SoVITS Main Model Config"
        sovits_diff_model_config_label = "SoVITS Diffusion Config"

    class preprocess(Locale.preprocess):
        tip = """
            Please place your dataset (i.e., a bunch of `.wav` files) in the `dataset_raw/character_name` folder under the main package directory.

            You can create multiple character folders to train different characters simultaneously.

            Your directory should look like this:

            ```
            dataset_raw/
            |-character_name1/
            |  | 1.wav
            |  | 2.wav
            |  | 3.wav
            |  ...
            |-character_name2/
            |  | 1.wav
            |  | 2.wav
            |  | 3.wav
            |  ...
            ```

            If you're unsure, simply click the button below to start automatic data processing.

            If you understand parameter meanings, switch to manual mode for more detailed processing.
            
            **CPU users should use FCPE as the F0 extractor/predictor.**
        """
        low_vram_tip = """
            ## Current device does not have a GPU with over 6GB of VRAM, recommended for training DDSP models only.

            Note: This doesn't mean you cannot train on other models!
        """

        open_dataset_folder_btn_value = "Open Dataset Folder"

        choose_model_label = "Select Model"
        start_preprocess_btn_value = "Start Preprocessing"

    class train(Locale.train):
        current_train_model_label = "Current Training Model"

        fouzu_tip = "~~Added a cyber Buddha, hoping it brings some help~~"

        gd_plus_1 = "Click to add 1 Karma"
        gd_plus_1_tip = "Karma +1, Burnout -1"

        choose_sub_model_label = "Select Sub-Model"
        choose_pretrain_model_label = "Select Pretrained Model"
        choose_pretrain_model_info = (
            "Choose a suitable base model according to your device. You can obtain more base models from the official website."
        )

        pretrain_model_vec = "Encoder"
        pretrain_model_vocoder = "Vocoder"
        pretrain_model_size = "Network Parameters"
        pretrain_model_attn = "Attention Mechanism"
        official_pretrain_model = "Official Pretrained Model"

        load_pretrained_failed_tip = (
            "Failed to load pretrained model, possible reasons are incompatible parameters or lack of a pretrained model."
        )

        start_train_btn_value = "Start/Continue Training"

        archive_btn_value = "Archive Working Directory"
        stop_btn_value = "Stop Training"

        archieving_tip = "Archiving in progress, do not click repeatedly."
        archived_tip = "Archived successfully, please check the folder."

        stopped_tip = "Training stop command sent, please check training window."

        tensorboard_btn = "Launch TensorBoard"

        launching_tb_tip = "Launching TensorBoard, please wait"
        launched_tb_tip = "TensorBoard is available at {1}"

    class settings(Locale.settings):
        page = "Page"

        save_btn_value = "Save Settings"

        pkg_settings_label = "Package Settings"
        infer_settings_label = "Inference Settings"
        sovits_settings_label = "So-VITS-SVC Settings"
        ddsp6_settings_label = "DDSP-SVC 6 Settings"
        ddsp6_1_settings_label = "DDSP-SVC 6.1 Settings"

        class pkg(Locale.settings.pkg):
            lang_label = "Language"
            lang_info = "Changing language requires restarting the package."

        class infer(Locale.settings.infer):
            msst_device_label = "Device for Separation Tasks"

        class sovits(Locale.settings.sovits):
            resolve_port_clash_label = "Attempt to Resolve Port Conflict (Windows Only)"

        class ddsp6(Locale.settings.ddsp6):
            pretrained_model_preference_dropdown_label = "Base Model Preference"
            default_pretrained_model = "Default Base Model 512 6"
            large_pretrained_model = "Large Network Base Model 1024 12"

        class ddsp6_1(Locale.settings.ddsp6_1):
            pretrained_model_preference_dropdown_label = "Base Model Preference"
            default_pretrained_model = "Default (Large Network) Base Model 1024 10"

        saved_tip = "Settings Saved"

    class install_model(Locale.install_model):
        tip = """
        ## Currently, only .sf_pkg/.h0_ddsp_pkg_model format model packages are supported.
        """

        file_label = "Upload Model Package"

        model_name_label = "Model Name"
        model_name_placeholder = "Please enter the model name"

        submit_btn_value = "Install Model"

    class path_chooser(Locale.path_chooser):
        input_path_label = "Input Folder"
        output_path_label = "Output Folder"

    class fish_audio_preprocess(Locale.fish_audio_preprocess):
        to_wav_tab = "Batch Convert WAV"
        slice_audio_tab = "Slicer"
        preprocess_tab = "Data Processing"
        max_duration_label = "Max Duration"
        submit_btn_value = "Start"

        input_output_same_tip = "Input and output paths are the same."
        input_path_not_exist_tip = "Input path does not exist."

    class vocal_separation(Locale.vocal_separation):
        input_audio_label = "Input Audio"
        input_path_label = "Input Path"
        output_path_label = "Output Path"

        use_batch_label = "Enable Batch Processing"
        use_de_reverb_label = "De-reverb"
        use_harmonic_remove_label = "Remove Harmonics"

        submit_btn_value = "Start"
        vocal_label = "Vocal Output"
        inst_label = "Instrumental Output"

        batch_output_message_label = "Batch Output Message"

        no_file_tip = "No file selected."
        no_input_tip = "Input folder not selected."
        no_output_tip = "Output folder not selected."
        input_not_exist_tip = "Input folder does not exist."
        output_not_exist_tip = "Output folder does not exist."
        input_output_same_tip = "Input and output folders are the same."

        finished = "Completed"
        error_when_processing = "Error during processing, please screenshot console for help"

        unusable_file_tip = "{1} skipped, unsupported file format"

        batch_progress_desc = "Total Progress"

        job_to_progress_desc = {
            "vocal": "Vocal Separation",
            "kim_vocal": "Vocal Separation",
            "deverb": "De-reverb",
            "karaoke": "Remove Harmonics",
        }

    class common_infer(Locale.common_infer):
        audio_label = "Audio File"

        use_batch_label = "Enable Batch Processing"

        use_vocal_separation_label = "Remove Instrumentals"
        use_vocal_separation_info = "Whether to remove instrumentals"

        use_de_reverb_label = "De-reverb"
        use_de_reverb_info = "Whether to de-reverb"

        use_harmonic_remove_label = "Remove Harmonics"
        use_harmonic_remove_info = "Whether to remove harmonics"

        f0_label = "F0 Extractor"
        f0_info = "Model for pitch extraction/prediction"

        keychange_label = "Key Change"
        keychange_info = "Reference: 12 for male-to-female, -12 for female-to-male. Adjust if voice doesn't sound right."

        threshold_label = "Threshold Value"
        threshold_info = "Vocal slice threshold, adjust to -40 or higher if background noise is present"

    class ddsp_based_infer(Locale.ddsp_based_infer):
        method_label = "Sampling Method"
        method_info = "Method for reflow sampling"

        infer_step_label = "Inference Step Size"
        infer_step_info = "Step size for inference, default setting works fine."

        t_start_label = "T Start"
        t_start_info = "Not specified"

        num_formant_shift_key_label = "Formant Shift Key"
        num_formant_shift_key_info = "Higher values make the voice thinner, lower values make it thicker"

    class ddsp_based_preprocess(Locale.ddsp_based_preprocess):
        method_label = "F0 Extractor"
        method_info = "Method for reflow sampling"

    class common_preprocess(Locale.common_preprocess):
        encoder_label = "Voice Encoder"
        encoder_info = "Model to encode the voice."

        f0_label = "F0 Extractor"
        f0_info = "Model for pitch extraction/prediction"

    class sovits(Locale.sovits):
        dataset_not_complete_tip = "Dataset incomplete, please check or reprocess."
        finished = "Completed"

        class train_main(Locale.sovits.train_main):
            log_interval_label = "Log Interval"
            log_interval_info = "Output a log every N steps."

            eval_interval_label = "Evaluation Interval"
            eval_interval_info = "Save and evaluate every N steps."

            all_in_mem_label = "Cache Full Dataset"
            all_in_mem_info = (
                "Load the entire dataset into memory for faster training, requires sufficient memory."
            )

            keep_ckpts_label = "Keep Checkpoints"
            keep_ckpts_info = "Keep recent N checkpoints."

            batch_size_label = "Training Batch Size"
            batch_size_info = "Larger is better, but consumes more VRAM."

            learning_rate_label = "Learning Rate"
            learning_rate_info = "Learning rate."

            num_workers_label = "Data Loader Workers"
            num_workers_info = "Enable if CPU cores > 4. More workers generally improve performance."

            half_type_label = "Precision"
            half_type_info = "Select fp16 for faster speed, but increases the risk of 'burnout'"

        class train_diff(Locale.sovits.train_diff):
            batchsize_label = "Training Batch Size"
            batchsize_info = "Larger is better, but consumes more VRAM. Ensure it does not exceed dataset size."

            num_workers_label = "Training Workers"
            num_workers_info = "Set to 0 if your GPU is powerful enough"

            amp_dtype_label = "Training Precision"
            amp_dtype_info = "Select fp16 or bf16 for faster speed, but increases risk of 'burnout'"

            lr_label = "Learning Rate"
            lr_info = "Not recommended to change."

            interval_val_label = "Validation Interval"
            interval_val_info = "Validate and save every N steps."

            interval_log_label = "Log Interval"
            interval_log_info = "Output a log every N steps."

            interval_force_save_label = "Force Save Model Interval"
            interval_force_save_info = "Save the model every N steps."

            gamma_label = "LR Decay Factor"
            gamma_info = "Not recommended to change."

            cache_device_label = "Cache Device"
            cache_device_info = "Select cuda for faster speed, but requires more VRAM (not applicable for SoVITS main models)"

            cache_all_data_label = "Cache All Data"
            cache_all_data_info = "Faster training with larger memory/VRAM"

            epochs_label = "Max Training Epochs"
            epochs_info = "Training stops when reaching this value."

            use_pretrain_label = "Use Pretrained Model"
            use_pretrain_info = (
                "Enabling can significantly reduce training time, leave unchanged if unsure."
            )

        class train_cluster(Locale.sovits.train_cluster):
            cluster_or_index_label = "Clustering or Retrieval"
            cluster_or_index_info = "Training clustering or retrieval models. Retrieval tends to be slightly better than clustering."

            use_gpu_label = "Use GPU"
            use_gpu_info = "Using a GPU accelerates training, only applicable for clustering"

        class infer(Locale.sovits.infer):
            cluster_infer_ratio_label = "Clustering/Feature Ratio"
            cluster_infer_ratio_info = (
                "Proportion of clustering or features, range 0-1. Set to 0 if no clustering model is trained."
            )

            linear_gradient_info = "Length of crossfade for two audio slices"
            linear_gradient_label = "Gradient Length"

            k_step_label = "Diffusion Steps"
            k_step_info = "Higher values result in more diffuse results, default is 100."

            enhancer_adaptive_key_label = "Enhancer Adaptation Key"
            enhancer_adaptive_key_info = "Makes the enhancer adapt to a wider pitch range (in half-steps). Default is 0."

            f0_filter_threshold_label = "F0 Filter Threshold"
            f0_filter_threshold_info = (
                "Effective when using crepe, lower values reduce tuning errors but increase silence."
            )

            audio_predict_f0_label = "Auto F0 Prediction"
            audio_predict_f0_info = (
                "Automatic pitch prediction for speech conversion. Disable this for singing voices to avoid severe pitch misalignment."
            )

            second_encoding_label = "Second Encoding"
            second_encoding_info = (
                "Performs a secondary encoding on the original audio before shallow diffusion, often has mixed results."
            )

            clip_label = "Force Slice Length"
            clip_info = "Force audio slice length, 0 for no forced slicing."

        class preprocess(Locale.sovits.preprocess):
            use_diff_label = "Train Shallow Diffusion"
            use_diff_info = (
                "Generate files required for training shallow diffusion when selected, slower than not selecting."
            )

            vol_aug_label = "Volume Embedding"
            vol_aug_info = "Enables volume embedding."

            num_workers_label = "Workers"
            num_workers_info = "Higher number is faster"

            subprocess_num_workers_label = "Subprocess Workers"
            subprocess_num_workers_info = "Higher number is faster"

            debug_label = "Enable Debug Mode"
            debug_info = "Output debugging information, not necessary for general use."

        class model_types(Locale.sovits.model_types):
            main = "Main Model"
            diff = "Shallow Diffusion"
            cluster = "Clustering/Retrieval Model"

        class model_chooser_extra(Locale.sovits.model_chooser_extra):
            enhance_label = "NSFHifigan Audio Enhancement"
            enhance_info = (
                "Improves audio quality for models with limited training data, may have negative effects on well-trained models."
            )

            feature_retrieval_label = "Enable Feature Retrieval"
            feature_retrieval_info = "Whether to use feature retrieval. Disabled if using clustering model."

            only_diffusion_label = "Shallow Diffusion Only"
            only_diffusion_info = "Only infer shallow diffusion, not recommended."

    class ddsp6(Locale.ddsp6):
        infer_tip = "Infer DDSP Model"

        class model_types(Locale.ddsp6.model_types):
            cascade = "Cascade Model"

        class train(Locale.ddsp6.train):
            batch_size_label = "Training Batch Size"
            batch_size_info = "Larger is better, but consumes more VRAM. Ensure it does not exceed dataset size."

            num_workers_label = "Training Workers"
            num_workers_info = "Set to 0 if your GPU is powerful enough"

            amp_dtype_label = "Training Precision"
            amp_dtype_info = "Select fp16 or bf16 for faster speed, but increases risk of 'burnout'"

            lr_label = "Learning Rate"
            lr_info = "Not recommended to change."

            interval_val_label = "Validation Interval"
            interval_val_info = "Validate and save every N steps."

            interval_log_label = "Log Interval"
            interval_log_info = "Output a log every N steps."

            interval_force_save_label = "Force Save Model Interval"
            interval_force_save_info = "Save the model every N steps."

            gamma_label = "LR Decay Factor"
            gamma_info = "Not recommended to change."

            cache_device_label = "Cache Device"
            cache_device_info = "Select cuda for faster speed, but requires more VRAM (not applicable for SoVITS main models)"

            cache_all_data_label = "Cache All Data"
            cache_all_data_info = "Faster training with larger memory/VRAM"

            epochs_label = "Max Training Epochs"
            epochs_info = "Training stops when reaching this value."

            use_pretrain_label = "Use Pretrained Model"
            use_pretrain_info = (
                "Enabling can significantly reduce training time, leave unchanged if unsure."
            )

    class reflow(Locale.reflow):
        infer_tip = "Infer ReflowVAESVC Model"

        class train(Locale.ddsp6.train):
            batch_size_label = "Training Batch Size"
            batch_size_info = "Larger is better, but consumes more VRAM. Ensure it does not exceed dataset size."

            num_workers_label = "Training Workers"
            num_workers_info = "Set to 0 if your GPU is powerful enough"

            amp_dtype_label = "Training Precision"
            amp_dtype_info = "Select fp16 or bf16 for faster speed, but increases risk of 'burnout'"

            lr_label = "Learning Rate"
            lr_info = "Not recommended to change."

            interval_val_label = "Validation Interval"
            interval_val_info = "Validate and save every N steps."

            interval_log_label = "Log Interval"
            interval_log_info = "Output a log every N steps."

            interval_force_save_label = "Force Save Model Interval"
            interval_force_save_info = "Save the model every N steps."

            gamma_label = "LR Decay Factor"
            gamma_info = "Not recommended to change."

            cache_device_label = "Cache Device"
            cache_device_info = "Select cuda for faster speed, but requires more VRAM (not applicable for SoVITS main models)"

            cache_all_data_label = "Cache All Data"
            cache_all_data_info = "Faster training with larger memory/VRAM"

            epochs_label = "Max Training Epochs"
            epochs_info = "Training stops when reaching this value."

            use_pretrain_label = "Use Pretrained Model"
            use_pretrain_info = (
                "Enabling can significantly reduce training time, leave unchanged if unsure."
            )

        class model_types(Locale.reflow.model_types):
            cascade = "Cascade Model"

    default_spk_name = "Default Speaker"

    preprocess_draw_desc = "Split Validation Set"
    preprocess_desc = "Preprocessing (check terminal for progress)"
    preprocess_finished = "Preprocessing Complete"
    