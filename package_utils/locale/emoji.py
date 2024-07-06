from package_utils.locale.base import Locale

locale_name = "emoji"
locale_display_name = "😎"


class _Locale(Locale):
    unknown_model_type_tip = "❓🤖, 🤔🔍"

    class device_chooser(Locale.device_chooser):
        device_dropdown_label = "📱💻"

    class model_chooser(Locale.model_chooser):
        submit_btn_value = "✔️🤖"
        model_type_dropdown_label = "🤖🔍"
        search_path_label = "🔍🛤️"
        workdir_name = "🛠️📁"
        archieve_dir_name = "📦📁"
        models_dir_name = "🤖📁"
        no_model_value = "❌🤖"
        unuse_value = "❌"
        no_spk_value = "❌🗣️"
        choose_model_dropdown_prefix = "✔️🤖"
        refresh_btn_value = "🔄🛠️"
        spk_dropdown_label = "✔️🗣️"
        no_spk_option = "❌🗣️"

    class form(Locale.form):
        submit_btn_value = "✔️"
        audio_output_1 = "🔊📈"
        audio_output_2 = "🔊📈/🎶"
        textbox_output = "📄📈"
        dorpdown_liked_checkbox_yes = "✔️"
        dorpdown_liked_checkbox_no = "❌"

    class model_manager(Locale.model_manager):
        pack_btn_value = "📦🤖"
        pack_result_label = "📦📈"
        packing_tip = "📦⏳, ❌🔄"
        unpackable_tip = "❌📦"
        clean_log_btn_value = "🗑️📝(✔️❌🛠️)"
        change_model_type_info = """
        #### 🔄🤖
        ❗❓🤖🔍！❌🔄🤖！✔️🔍🤖！
        """
        change_model_type_btn_value = "✔️🔄"
        change_success_tip = "✔️🔄"
        change_fail_tip = "❌🔄"

    class main_ui(Locale.main_ui):
        release_memory_btn_value = "🛠️💾/🧠"
        released_tip = "✔️🛠️💾/🧠"
        infer_tab = "💡🛠️"
        preprocess_tab = "⏳🛠️"
        train_tab = "🏋️\u200d♂️🛠️"
        tools_tab = "🛠️🔧"
        settings_tab = "🪡⚙️"
        model_manager_tab = "🤖🛠️"
        install_model_tab = "⬇️🤖"
        fish_audio_preprocess_tab = "🔄🎵"
        vocal_remove_tab = "🗣️❌"
        detect_spk_tip = "✔️🗣️:"
        spk_not_found_tip = "❌🗣️"

    class preprocess(Locale.preprocess):
        tip = """
            ✔️📂(`.wav` 📂) 📂 `dataset_raw/🗣️📛`

            ✔️📂🆕🗣️📂✔️🛠️🆕🗣️

            ✔️📂🔍:

            ```
            dataset_raw/
            |-🗣️1/
            |  | 1.wav
            |  | 2.wav
            |  | 3.wav
            |  ...
            |-🗣️2/
            |  | 1.wav
            |  | 2.wav
            |  | 3.wav
            |  ...
            ```

            ❓, ✔️🔘⬇️📄🔄🛠️

            ✔️🔄✔️, ✔️🛠️🔧🔄🛠️
            
            **CPU 🔧✔️ FCPE F0 🔍/🔮**
        """
        little_vram_tip = """
            ## 💻❌🗄️ 6GB, ✔️🛠️ DDSP 🤖
        """
        open_dataset_folder_btn_value = "📂📂"
        choose_model_label = "✔️🤖"

    class train(Locale.train):
        current_train_model_label = "🔄🤖"
        gd_plus_1 = "✔️🛠️"
        gd_plus_1_tip = "🛠️ +1, 🔄 -1"
        fouzu_tip = "👤🙏🧘\u200d♂️✨"
        choose_sub_model_label = "✔️🤖"
        archieve_btn_value = "📦📁"
        stop_btn_value = "❌🔄"
        archieving_tip = "📦⏳, ❌🔄"
        archieved_tip = "✔️📦, 📂📂"
        stopped_tip = "✔️❌🔄, 🔄📄"

    class settings(Locale.settings):
        page = "📄"
        pkg_settings_label = "📦⚙️"
        sovits_settings_label = "SoVITS ⚙️"

        class pkg(Locale.settings.pkg):
            lang_label = "🈯"
            lang_info = "🔄🈯✔️🔄📦"

        class sovits(Locale.settings.sovits):
            resolve_port_clash_label = "🔄🔧🛤️"

        saved_tip = "✔️"
        ddsp6_settings_label = ""

    class install_model(Locale.install_model):
        tip = """
        ## ✔️⬆️ .sf_pkg/.h0_ddsp_pkg_model 📦
        """
        file_label = "⬆️📦"
        model_name_label = "🤖📛"
        model_name_placeholder = "🔤🤖📛"
        submit_btn_value = "⬇️🤖"

    class path_chooser(Locale.path_chooser):
        input_path_label = "📂"
        output_path_label = "📂"

    class fish_audio_preprocess(Locale.fish_audio_preprocess):
        to_wav_tab = "🔄 WAV"
        slice_audio_tab = "✂️🎵"
        preprocess_tab = "🛠️🔄"
        max_duration_label = "⏳📏"
        submit_btn_value = "▶️"
        input_path_not_exist_tip = "❌📂"

    class vocal_remove(Locale.vocal_remove):
        input_audio_label = "🎵"
        submit_btn_value = "▶️"
        vocal_label = "🔄-🗣️"
        inst_label = "🔄-🎶"

    class common_infer(Locale.common_infer):
        audio_label = "🎵"
        use_vocal_remove_label = "🗣️❌"
        use_vocal_remove_info = "❓🗣️❌"
        f0_label = "f0 🔍"
        f0_info = "🎵🔍/🔮🤖"
        keychange_label = "🔄🎵"
        keychange_info = "🔄: 👨🎵👩 12, 👩🎵👨 -12, 🔄🔄"
        threshold_label = "✂️📉"
        threshold_info = "🗣️✂️📉, ❗️📉🔊 ✔️ -40 🔄⬆️"

    class diff_based_infer(Locale.diff_based_infer):
        method_label = "🔍"
        method_info = "🔄 reflow 🔍"
        infer_step_label = "🛠️⏳"
        infer_step_info = "🛠️📏, ✔️🔧"
        t_start_label = "T ⏳"
        t_start_info = "❓"

    class diff_based_preprocess(Locale.diff_based_preprocess):
        method_label = "f0 🔍"
        method_info = "🔄 reflow 🔍"

    class common_preprocess(Locale.common_preprocess):
        encoder_label = "🗣️🔍"
        encoder_info = "🗣️🔍🤖"
        f0_label = "f0 🔍"
        f0_info = "🎵🔍/🔮🤖"

    class sovits(Locale.sovits):
        dataset_not_complete_tip = "📂❌, 🔍📂🔄🔄"
        finished = "✔️"

        class infer(Locale.sovits.infer):
            cluster_infer_ratio_label = "🗣️/📝📉"
            cluster_infer_ratio_info = "🗣️/📝📉, 📏0-1, ❌🗣️/🔍✔️0"
            linear_gradient_info = "🔄🎵✂️📏"
            linear_gradient_label = "✂️📏"
            k_step_label = "⏳📏"
            k_step_info = "⬆️⬆️🔄🤖, ✔️100"
            enhancer_adaptive_key_label = "🔄🔧"
            enhancer_adaptive_key_info = "🔧⬆️🎵🔝(🔤⬆️)|✔️0"
            f0_filter_threshold_label = "f0 ✋📉"
            f0_filter_threshold_info = "✔️ crepe❗️. 📏0-1. ⬇️📉❓⬇️🔄, ⬆️🔇"
            audio_predict_f0_label = "🔮 f0"
            audio_predict_f0_info = "🗣️🔄🔮🎵, 🎤❌🔄⬇️⬇️📉"
            second_encoding_label = "2️⃣🔍"
            second_encoding_info = "🔍🎵2️⃣🔍, 🔮, ✔️⬆️, ❌⬆️"
            clip_label = "✂️📏📏"
            clip_info = "✂️🎵📏📏, 0 ❌✂️"

        class preprocess(Locale.sovits.preprocess):
            use_diff_label = "🛠️🔄"
            use_diff_info = "✔️🔄🛠️🛠️, ❌⬇️"
            vol_aug_label = "🔊🔧"
            vol_aug_info = "✔️🔊🔧"
            num_workers_label = "🔧📏"
            num_workers_info = "⬆️⬆️🔄⬆️"
            subprocess_num_workers_label = "🔧📏🔩"
            subprocess_num_workers_info = "⬆️⬆️🔄⬆️"

        class model_types(Locale.sovits.model_types):
            main = "🤖"
            diff = "🛠️🔄"
            cluster = "🔍"

        class model_chooser_extra(Locale.sovits.model_chooser_extra):
            enhance_label = "NSFHifigan 🔧🎵"
            enhance_info = "📂❌🤖⬆️🎵🔧, 📂✔️🤖⬇️⬇️🔧"
            feature_retrieval_label = "🔧🔍"
            feature_retrieval_info = "❓🔧🔍, ✔️🔍❌"

        train_main = ""
        train_diff = ""
        train_cluster = ""

    class ddsp6(Locale.ddsp6):
        infer_tip = "🛠️ DDSP 🤖"

        class model_types(Locale.ddsp6.model_types):
            cascade = "🔄🤖"

        class train(Locale.ddsp6.train):
            batch_size_label = "🛠️📏"
            batch_size_info = "⬆️⬆️, ⬆️⬆️🗄️, ❗️❌📂📏"
            num_workers_label = "🛠️🔧📏"
            num_workers_info = "💻✔️, ✔️0"
            amp_dtype_label = "🛠️📈"
            amp_dtype_info = "✔️ fp16、bf16 ⬆️⬆️, ❗️🔄⬆️⬆️"
            lr_label = "🔄⬆️"
            lr_info = "❌🔄"
            interval_val_label = "⏳📏"
            interval_val_info = "N ⏳, ✔️💾"
            interval_log_label = "📝📏"
            interval_log_info = "N 📝📄"
            interval_force_save_label = "✔️💾📏"
            interval_force_save_info = "N 💾🤖"
            gamma_label = "🔄📉"
            gamma_info = "❌🔄"
            cache_device_label = "🛠️🖥️"
            cache_device_info = "✔️ cuda ⬆️⬆️, ❗️⬆️🗄️(SoVITS 🤖❌)"
            cache_all_data_label = "🛠️🗄️"
            cache_all_data_info = "⬆️⬆️, ❗️⬆️🗄️/🗄️"
            epochs_label = "🔄⏳📏"
            epochs_info = "✔️🔄📏⬇️"
            use_pretrain_label = "✔️🔄🤖"
            use_pretrain_info = "✔️⬇️🔄🛠️, ❓❌🔄"

    class reflow(Locale.reflow):
        infer_tip = "🛠️ ReflowVAESVC 🤖"

        class train(Locale.ddsp6.train):
            batch_size_label = "🛠️📏"
            batch_size_info = "⬆️⬆️, ⬆️⬆️🗄️, ❗️❌📂📏"
            num_workers_label = "🛠️🔧📏"
            num_workers_info = "💻✔️, ✔️0"
            amp_dtype_label = "🛠️📈"
            amp_dtype_info = "✔️ fp16、bf16 ⬆️⬆️, ❗️🔄⬆️⬆️"
            lr_label = "🔄⬆️"
            lr_info = "❌🔄"
            interval_val_label = "⏳📏"
            interval_val_info = "N ⏳, ✔️💾"
            interval_log_label = "📝📏"
            interval_log_info = "N 📝📄"
            interval_force_save_label = "✔️💾📏"
            interval_force_save_info = "N 💾🤖"
            gamma_label = "🔄📉"
            gamma_info = "❌🔄"
            cache_device_label = "🛠️🖥️"
            cache_device_info = "✔️ cuda ⬆️⬆️, ❗️⬆️🗄️(SoVITS 🤖❌)"
            cache_all_data_label = "🛠️🗄️"
            cache_all_data_info = "⬆️⬆️, ❗️⬆️🗄️/🗄️"
            epochs_label = "🔄⏳📏"
            epochs_info = "✔️🔄📏⬇️"
            use_pretrain_label = "✔️🔄🤖"
            use_pretrain_info = "✔️⬇️🔄🛠️, ❓❌🔄"

        class model_types(Locale.reflow.model_types):
            cascade = "🔄🤖"

    default_spk_name = "✔️🗣️"
    preprocess_draw_desc = "✂️🔍"
    preprocess_desc = "🛠️🔍(📈🔍🖥️)"
    preprocess_finished = "✔️🔍"
