from .base import Locale


class enUSLocale(Locale):
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

        choose_model_dropdown_prefix = "Select Model"

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

        clean_log_btn_value = "Clear Logs (confirm no more training before clearing)"

        change_model_type_info = """
        #### Change Model Type
        Use only when the model type cannot be recognized! This is not for converting model types! It's for changing the recognized model type!
        """
        change_model_type_btn_value = "Confirm Change"
        change_success_tip = "Change Successful"
        change_fail_tip = "Change Failed"

    class main_ui(Locale.main_ui):
        release_memory_btn_value = "Attempt to Release VRAM/Memory"
        released_tip = "Attempted to Release VRAM/Memory"
        infer_tab = "💡Inference"
        preprocess_tab = "⏳Data Processing"
        train_tab = "🏋️‍♂️Training"
        tools_tab = "🛠️Tools"
        settings_tab = "🪡Settings"

        model_manager_tab = "Model Management"
        install_model_tab = "Install Model"
        fish_audio_preprocess_tab = "Simple Audio Processing"
        vocal_remove_tab = "Vocal Separation"

        detect_spk_tip = "Detected Characters:"
        spk_not_found_tip = "No Characters Detected"

    class preprocess(Locale.preprocess):
        tip = """
            Please first place your dataset (i.e., a bunch of `.wav` files) into the `dataset_raw/your_character_name` folder in the package

            You can train multiple characters at the same time by creating multiple character folders

            After placing, your directory should look like this

            ```
            dataset_raw/
            |-your_character_name1/
            |  | 1.wav
            |  | 2.wav
            |  | 3.wav
            |  ...
            |-your_character_name2/
            |  | 1.wav
            |  | 2.wav
            |  | 3.wav
            |  ...
            ```

            If you don't understand anything, just click the button below for fully automated data processing

            If you understand the significance of the parameters, you can switch to manual mode for more detailed data processing
            
            **CPU users, please use FCPE as the F0 extractor/predictor**
        """
        little_vram_tip = """
            ## The current device does not have a graphics card with more than 6GB of VRAM, it is recommended to only train DDSP models
        """

        open_dataset_folder_btn_value = "Open Dataset Folder"

        choose_model_label = "Select Model"

    class settings(Locale.settings):
        pkg_settings_label = "Package Settings"

        lang_label = "Language"
        lang_info = "Changing the language requires restarting the package"

    class install_model(Locale.install_model):
        tip = """
        ## Currently only supports uploading .sf_pkg/.h0_ddsp_pkg_model format model packages
        """

        file_label = "Upload Model Package"

        model_name_label = "Model Name"
        model_name_placeholder = "Please enter model name"

        submit_btn_value = "Install Model"

    class path_chooser(Locale.path_chooser):
        input_path_label = "Input Folder"
        output_path_label = "Output Folder"

    class fish_audio_preprocess(Locale.fish_audio_preprocess):
        to_wav_tab = "Batch Convert to WAV"
        slice_audio_tab = "Audio Slicer"
        preprocess_tab = "Data Processing"
        max_duration_label = "Max Duration"
        submit_btn_value = "Start"

        input_path_not_exist_tip = "Input path does not exist"

    class vocal_remove(Locale.vocal_remove):
        input_audio_label = "Input Audio"
        submit_btn_value = "Start"
        vocal_label = "Output - Vocals"
        inst_label = "Output - Instrumental"

    class sovits(Locale.sovits):
        dataset_not_complete_tip = (
            "Dataset is incomplete, please check the dataset or preprocess again"
        )
        finished = "Completed"

        class infer(Locale.sovits.infer):
            cluster_infer_ratio_label = "Cluster/Feature Ratio"
            cluster_infer_ratio_info = "Ratio of clustering/features, range 0-1. If no clustering model or feature retrieval is trained, default to 0"

            linear_gradient_info = "Crossfade length for two audio segments"
            linear_gradient_label = "Gradient Length"

            k_step_label = "Diffusion Steps"
            k_step_info = "The larger the number, the closer to the diffusion model result, default is 100"

            enhancer_adaptive_key_label = "Enhancer Adaptation"
            enhancer_adaptive_key_info = "Make the enhancer adapt to a higher pitch range (unit is semitone)|Default is 0"

            f0_filter_threshold_label = "f0 Filter Threshold"
            f0_filter_threshold_info = "Only effective when using crepe. The range is from 0-1. Lowering this value can reduce the probability of off-pitch, but increase the probability of muted sound"

            audio_predict_f0_label = "Automatic f0 Prediction"
            audio_predict_f0_info = "Automatically predict pitch for voice conversion, do not turn this on when converting singing voice, it will cause severe off-pitch"

            second_encoding_label = "Secondary Encoding"
            second_encoding_info = "Secondary encoding of the original audio before shallow diffusion, esoteric option, sometimes it works well, sometimes it works poorly"

            clip_label = "Force Clip Length"
            clip_info = "Force the audio clip length, 0 means no force"

        class train(Locale.sovits.train):
            use_diff_label = "Train Shallow Diffusion"
            use_diff_info = "Check this to generate files needed for training shallow diffusion, it will be slower than not selecting"

            vol_aug_label = "Volume Embedding"
            vol_aug_info = "Check this to use volume embedding"

            num_workers_label = "Number of Processes"
            num_workers_info = "Theoretically the larger the number, the faster"

            subprocess_num_workers_label = "Number of Threads per Process"
            subprocess_num_workers_info = (
                "Theoretically the larger the number, the faster"
            )

        class model_types(Locale.sovits.model_types):
            main = "Main Model"
            diff = "Shallow Diffusion"
            cluster = "Clustering/Retrieval Model"

        class model_chooser_extra(Locale.sovits.model_chooser_extra):
            enhance_label = "NSFHifigan Audio Enhancement"
            enhance_info = "Has certain audio quality enhancement effects for models with a small training set, has adverse effects on well-trained models"

            feature_retrieval_label = "Enable Feature Retrieval"
            feature_retrieval_info = "Whether to use feature retrieval, will be disabled if clustering model is used"
