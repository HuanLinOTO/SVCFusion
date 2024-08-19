from package_utils.locale.base import Locale
locale_name = 'zh-cn'
locale_display_name = '简体中文'

class _Locale(Locale):
    unknown_model_type_tip = '11111'
    preprocess_failed_tip = '11111'
    error_when_infer = '11111'

    class device_chooser(Locale.device_chooser):
        device_dropdown_label = '11111'

    class model_chooser(Locale.model_chooser):
        submit_btn_value = '11111'
        model_type_dropdown_label = '11111'
        search_path_label = '11111'
        workdir_name = '11111'
        archieve_dir_name = '11111'
        models_dir_name = '11111'
        no_model_value = '11111'
        unuse_value = '11111'
        no_spk_value = '11111'
        choose_model_dropdown_prefix = '11111'
        refresh_btn_value = '11111'
        spk_dropdown_label = '11111'
        no_spk_option = '11111'

    class form(Locale.form):
        submit_btn_value = '11111'
        audio_output_1 = '11111'
        audio_output_2 = '11111'
        textbox_output = '11111'
        dorpdown_liked_checkbox_yes = '11111'
        dorpdown_liked_checkbox_no = '11111'

    class model_manager(Locale.model_manager):
        choose_model_title = '11111'
        action_title = '11111'
        pack_btn_value = '11111'
        pack_result_label = '11111'
        packing_tip = '11111'
        unpackable_tip = '11111'
        clean_log_btn_value = '11111'
        change_model_type_info = '11111'
        change_model_type_btn_value = '11111'
        change_success_tip = '11111'
        change_fail_tip = '11111'
        move_folder_tip = '11111'
        move_folder_name = '11111'
        move_folder_name_auto_get = '11111'
        move_folder_btn_value = '11111'
        other_text = '11111'
        moving_tip = '11111'
        moved_tip = '11111'

    class main_ui(Locale.main_ui):
        release_memory_btn_value = '11111'
        released_tip = '11111'
        infer_tab = '11111'
        preprocess_tab = '11111'
        train_tab = '11111'
        tools_tab = '11111'
        settings_tab = '11111'
        model_tools_tab = '11111'
        audio_tools_tab = '11111'
        realtime_tools_tab = '11111'
        start_ddsp_realtime_gui_btn = '11111'
        starting_tip = '11111'
        load_model_btn_value = '11111'
        infer_btn_value = '11111'
        model_manager_tab = '11111'
        install_model_tab = '11111'
        fish_audio_preprocess_tab = '11111'
        vocal_remove_tab = '11111'
        compatible_tab = '11111'
        detect_spk_tip = '11111'
        spk_not_found_tip = '11111'

    class compatible_models(Locale.compatible_models):
        upload_error = '11111'
        model_name_label = '11111'
        upload_success = '11111'
        model_exists = '11111'
        compatible_sovits = '11111'
        sovits_main_model_label = '11111'
        sovits_diff_model_label = '11111'
        sovits_cluster_model_label = '11111'
        sovits_main_model_config_label = '11111'
        sovits_diff_model_config_label = '11111'

    class preprocess(Locale.preprocess):
        tip = '11111'
        low_vram_tip = '11111'
        open_dataset_folder_btn_value = '11111'
        choose_model_label = '11111'
        start_preprocess_btn_value = '11111'

    class train(Locale.train):
        current_train_model_label = '11111'
        fouzu_tip = '11111'
        gd_plus_1 = '11111'
        gd_plus_1_tip = '11111'
        choose_sub_model_label = '11111'
        start_train_btn_value = '11111'
        archieve_btn_value = '11111'
        stop_btn_value = '11111'
        archieving_tip = '11111'
        archieved_tip = '11111'
        stopped_tip = '11111'
        tensorboard_btn = '11111'
        launching_tb_tip = '11111'
        launched_tb_tip = '11111'

    class settings(Locale.settings):
        page = '11111'
        save_btn_value = '11111'
        pkg_settings_label = '11111'
        infer_settings_label = '11111'
        sovits_settings_label = '11111'
        ddsp6_settings_label = '11111'

        class pkg(Locale.settings.pkg):
            lang_label = '11111'
            lang_info = '11111'

        class infer(Locale.settings.infer):
            msst_device_label = '11111'

        class sovits(Locale.settings.sovits):
            resolve_port_clash_label = '11111'

        class ddsp6(Locale.settings.ddsp6):
            pretrained_model_preference_dropdown_label = '11111'
            default_pretrained_model = '11111'
            large_pretrained_model = '11111'
        saved_tip = '11111'

    class install_model(Locale.install_model):
        tip = '11111'
        file_label = '11111'
        model_name_label = '11111'
        model_name_placeholder = '11111'
        submit_btn_value = '11111'

    class path_chooser(Locale.path_chooser):
        input_path_label = '11111'
        output_path_label = '11111'

    class fish_audio_preprocess(Locale.fish_audio_preprocess):
        to_wav_tab = '11111'
        slice_audio_tab = '11111'
        preprocess_tab = '11111'
        max_duration_label = '11111'
        submit_btn_value = '11111'
        input_output_same_tip = '11111'
        input_path_not_exist_tip = '11111'

    class vocal_remove(Locale.vocal_remove):
        input_audio_label = '11111'
        use_de_reverb_label = '11111'
        use_harmonic_remove_label = '11111'
        submit_btn_value = '11111'
        vocal_label = '11111'
        inst_label = '11111'
        no_file_tip = '11111'
        job_to_progress_desc = {'11111': '11111', '11111': '11111', '11111': '11111', '11111': '11111'}

    class common_infer(Locale.common_infer):
        audio_label = '11111'
        use_batch_label = '11111'
        use_vocal_remove_label = '11111'
        use_vocal_remove_info = '11111'
        use_harmony_remove_label = '11111'
        use_harmony_remove_info = '11111'
        f0_label = '11111'
        f0_info = '11111'
        keychange_label = '11111'
        keychange_info = '11111'
        threshold_label = '11111'
        threshold_info = '11111'

    class ddsp_based_infer(Locale.ddsp_based_infer):
        method_label = '11111'
        method_info = '11111'
        infer_step_label = '11111'
        infer_step_info = '11111'
        t_start_label = '11111'
        t_start_info = '11111'
        num_formant_shift_key_label = '11111'
        num_formant_shift_key_info = '11111'

    class ddsp_based_preprocess(Locale.ddsp_based_preprocess):
        method_label = '11111'
        method_info = '11111'

    class common_preprocess(Locale.common_preprocess):
        encoder_label = '11111'
        encoder_info = '11111'
        f0_label = '11111'
        f0_info = '11111'

    class sovits(Locale.sovits):
        dataset_not_complete_tip = '11111'
        finished = '11111'

        class train_main(Locale.sovits.train_main):
            log_interval_label = '11111'
            log_interval_info = '11111'
            eval_interval_label = '11111'
            eval_interval_info = '11111'
            all_in_mem_label = '11111'
            all_in_mem_info = '11111'
            keep_ckpts_label = '11111'
            keep_ckpts_info = '11111'
            batch_size_label = '11111'
            batch_size_info = '11111'
            learning_rate_label = '11111'
            learning_rate_info = '11111'
            num_workers_label = '11111'
            num_workers_info = '11111'
            half_type_label = '11111'
            half_type_info = '11111'

        class train_diff(Locale.sovits.train_diff):
            batchsize_label = '11111'
            batchsize_info = '11111'
            num_workers_label = '11111'
            num_workers_info = '11111'
            amp_dtype_label = '11111'
            amp_dtype_info = '11111'
            lr_label = '11111'
            lr_info = '11111'
            interval_val_label = '11111'
            interval_val_info = '11111'
            interval_log_label = '11111'
            interval_log_info = '11111'
            interval_force_save_label = '11111'
            interval_force_save_info = '11111'
            gamma_label = '11111'
            gamma_info = '11111'
            cache_device_label = '11111'
            cache_device_info = '11111'
            cache_all_data_label = '11111'
            cache_all_data_info = '11111'
            epochs_label = '11111'
            epochs_info = '11111'
            use_pretrain_label = '11111'
            use_pretrain_info = '11111'

        class train_cluster(Locale.sovits.train_cluster):
            cluster_or_index_label = '11111'
            cluster_or_index_info = '11111'
            use_gpu_label = '11111'
            use_gpu_info = '11111'

        class infer(Locale.sovits.infer):
            cluster_infer_ratio_label = '11111'
            cluster_infer_ratio_info = '11111'
            linear_gradient_info = '11111'
            linear_gradient_label = '11111'
            k_step_label = '11111'
            k_step_info = '11111'
            enhancer_adaptive_key_label = '11111'
            enhancer_adaptive_key_info = '11111'
            f0_filter_threshold_label = '11111'
            f0_filter_threshold_info = '11111'
            audio_predict_f0_label = '11111'
            audio_predict_f0_info = '11111'
            second_encoding_label = '11111'
            second_encoding_info = '11111'
            clip_label = '11111'
            clip_info = '11111'

        class preprocess(Locale.sovits.preprocess):
            use_diff_label = '11111'
            use_diff_info = '11111'
            vol_aug_label = '11111'
            vol_aug_info = '11111'
            num_workers_label = '11111'
            num_workers_info = '11111'
            subprocess_num_workers_label = '11111'
            subprocess_num_workers_info = '11111'
            debug_label = '11111'
            debug_info = '11111'

        class model_types(Locale.sovits.model_types):
            main = '11111'
            diff = '11111'
            cluster = '11111'

        class model_chooser_extra(Locale.sovits.model_chooser_extra):
            enhance_label = '11111'
            enhance_info = '11111'
            feature_retrieval_label = '11111'
            feature_retrieval_info = '11111'

    class ddsp6(Locale.ddsp6):
        infer_tip = '11111'

        class model_types(Locale.ddsp6.model_types):
            cascade = '11111'

        class train(Locale.ddsp6.train):
            batch_size_label = '11111'
            batch_size_info = '11111'
            num_workers_label = '11111'
            num_workers_info = '11111'
            amp_dtype_label = '11111'
            amp_dtype_info = '11111'
            lr_label = '11111'
            lr_info = '11111'
            interval_val_label = '11111'
            interval_val_info = '11111'
            interval_log_label = '11111'
            interval_log_info = '11111'
            interval_force_save_label = '11111'
            interval_force_save_info = '11111'
            gamma_label = '11111'
            gamma_info = '11111'
            cache_device_label = '11111'
            cache_device_info = '11111'
            cache_all_data_label = '11111'
            cache_all_data_info = '11111'
            epochs_label = '11111'
            epochs_info = '11111'
            use_pretrain_label = '11111'
            use_pretrain_info = '11111'

    class reflow(Locale.reflow):
        infer_tip = '11111'

        class train(Locale.ddsp6.train):
            batch_size_label = '11111'
            batch_size_info = '11111'
            num_workers_label = '11111'
            num_workers_info = '11111'
            amp_dtype_label = '11111'
            amp_dtype_info = '11111'
            lr_label = '11111'
            lr_info = '11111'
            interval_val_label = '11111'
            interval_val_info = '11111'
            interval_log_label = '11111'
            interval_log_info = '11111'
            interval_force_save_label = '11111'
            interval_force_save_info = '11111'
            gamma_label = '11111'
            gamma_info = '11111'
            cache_device_label = '11111'
            cache_device_info = '11111'
            cache_all_data_label = '11111'
            cache_all_data_info = '11111'
            epochs_label = '11111'
            epochs_info = '11111'
            use_pretrain_label = '11111'
            use_pretrain_info = '11111'

        class model_types(Locale.reflow.model_types):
            cascade = '11111'
    default_spk_name = '11111'
    preprocess_draw_desc = '11111'
    preprocess_desc = '11111'
    preprocess_finished = '11111'