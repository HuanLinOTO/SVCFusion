from SVCFusion.locale.base import Locale

locale_name = "emojilang"
locale_display_name = "😎"


class _Locale(Locale):
    unknown_model_type_tip = "🔍🔍🤖💡📖🌐🔄🔍🔍"
    preprocess_failed_tip = "👋🚫❗️🔍🖥📷💬"
    error_when_infer = "👋🏼"

    class device_chooser(Locale.device_chooser):
        device_dropdown_label = "📱"

    class model_chooser(Locale.model_chooser):
        submit_btn_value = "🤖🔍"
        model_type_dropdown_label = " Modelo tipo 🚨"
        search_path_label = "🔍🔄"
        workdir_name = "📁🚀"
        archive_dir_name = "📝📚🔒"
        models_dir_name = "📦 Modelo"
        no_model_value = "🚫 Modelo no encontrado"
        unuse_value = "🚫"
        no_spk_value = ":no_speech_bubbles:"
        choose_model_dropdown_prefix = "🤖🔍"
        refresh_btn_value = "🔄'options'"
        spk_dropdown_label = "👋🏼"
        no_spk_option = "🔍🤖"

    class form(Locale.form):
        submit_btn_value = "👋"
        audio_output_1 = "🌍🔍💡"
        audio_output_2 = "演奏音乐🎶 提供支持🎵"
        textbox_output = "🌍🔍💡"
        dorpdown_liked_checkbox_yes = "👋🏼"
        dorpdown_liked_checkbox_no = ":no_entry_sign:"

    class model_manager(Locale.model_manager):
        choose_model_title = "🤖🔍"
        action_title = "🤖🚀"
        pack_btn_value = "📦🤖"
        pack_result_label = "📦结果显示"
        packing_tip = "🔄📦，请🚫重复クリック"
        unpackable_tip = "🚫📦🤖"
        clean_log_btn_value = "🔄 Logs Reset (Confirm No More Training Before Reset)"
        change_model_type_info = "🔄 🎨 💪🏼 🤔 🔍 ➕ 🔑 📜"
        change_model_type_btn_value = "🤔👍🏼🔄"
        change_success_tip = "👍🏼✅"
        change_fail_tip = "🚫 تحديث проваленный"
        move_folder_tip = "🔄 ➕ 📂移到️ 🏷️`models`"
        move_folder_name = "🤖📝"
        move_folder_name_auto_get = "🤖📚🔍"
        move_folder_btn_value = " telefon"
        other_text = "รอ"
        moving_tip = "🔄🚫🙅"
        moved_tip = "🔄➡️👉🏼 `{1}`"

    class main_ui(Locale.main_ui):
        release_memory_btn_value = "🔄🖥️%/ данными"
        released_tip = "🔄🔍 vidéoramă"
        infer_tab = "💡🔎"
        preprocess_tab = "🔄🧮"
        train_tab = "🏋️\u200d♂️💪"
        tools_tab = "🛠️🧰"
        settings_tab = "🪡🛠️"
        model_tools_tab = "🤖🔗"
        audio_tools_tab = "🎶🎵🎧🎧📢🗣️🎤🎧"
        realtime_tools_tab = "ライブ"
        start_ddsp_realtime_gui_btn = "👋🚀📚🌐📊💰⏰💻📈🔍"
        starting_tip = "🔄🚀잠시후,다시클릭하지마세요.중대한결과가있습니다"
        load_model_btn_value = "🔄🤖 Modelo️"
        infer_btn_value = "💡🔍"
        model_manager_tab = "🤖 Quản lý"
        install_model_tab = "💡🤖🔍 Modelo de Instalación"
        fish_audio_preprocess_tab = "演奏🎶，简化обработка🎵"
        vocal_separation_tab = "🎶🎧"
        compatible_tab = " Modelo Compatible"
        detect_spk_tip = "👋🏼"
        spk_not_found_tip = "🔍🤖"

    class compatible_models(Locale.compatible_models):
        upload_error = "📦🚫➡️🔍📝✅"
        model_name_label = "🤖📝"
        upload_success = ".Upload réussi"
        model_exists = "🔍💡"
        compatible_sovits = "🤖🎵📈"
        sovits_main_model_label = " Modelo_principal_de_SOVITS"
        sovits_diff_model_label = "👨\u200d🎤💡🔄👩\u200d💻🔍"
        sovits_cluster_model_label = "🤖🔍"
        sovits_main_model_config_label = "🤖📝"
        sovits_diff_model_config_label = "соло 🌐 🔍💡"

    class preprocess(Locale.preprocess):
        tip = "👋🏻\n📝 📝 🇯́其他国家的输入法"
        low_vram_tip = "👋🏼\n\n## 📲:no_smoking: 🔢GPU内存容量,当前设备上没有任何一个大于6GB的显卡显存。我们仅推荐您在进行DDSP模型的训练时使用。  \n\n📚:warning: 注意：这并不意味着你无法进行训练！"
        open_dataset_folder_btn_value = '👋🌍🔍"data" 🗂️'
        choose_model_label = "🤖🔍"
        start_preprocess_btn_value = "🔄准备工作"

    class train(Locale.train):
        current_train_model_label = " Modelo de entrenamiento actual"
        fouzu_tip = "👋🚀🙏✨"
        gd_plus_1 = "🤔"
        gd_plus_1_tip = " cooker爆炸='-1',功德增加='+'"
        choose_sub_model_label = "🔍🤖"
        start_train_btn_value = "📚🚀🏃\u200d♂️🔥🔄🔄"
        archive_btn_value = "📚🔍폴더"
        stop_btn_value = "🚫🔥🤖️프로그래밍"
        archieving_tip = "🔍📚🔒🚫"
        archived_tip = "🗂️✅🔍폴더 열어서 확인해 주세요"
        stopped_tip = "👋🌍 ➡️👤🤖🔍📚👀🌐"
        tensorboard_btn = "🔥🚀💡"
        launching_tb_tip = "🚀🔍📝"
        launched_tb_tip = "🔍📚🌐💰"

    class settings(Locale.settings):
        page = "📖🌍"
        save_btn_value = "📌👍🏼📝"
        pkg_settings_label = "捆绑包设置"
        infer_settings_label = "🔍🛠️"
        sovits_settings_label = "👋🚫💡🛠️"
        ddsp6_settings_label = "🎤📚 seis"

        class pkg(Locale.settings.pkg):
            lang_label = "👋🏾"
            lang_info = "🔄🔧🌍"

        class infer(Locale.settings.infer):
            msst_device_label = "🏃🏽\u200d♂️🔍⚙️🔍📱"

        class sovits(Locale.settings.sovits):
            resolve_port_clash_label = "🔄🛠️💻🚀🚫🌐Mbps"

        class ddsp6(Locale.settings.ddsp6):
            pretrained_model_preference_dropdown_label = "🔍👌🏼"
            default_pretrained_model = " résult: ⚙️，默认尺寸：512×6"
            large_pretrained_model = "🔍🌐尺寸：1024×12"

        class ddsp6_1(Locale.settings.ddsp6_1):
            pretrained_model_preference_dropdown_label = "🔍👌🏼"
            default_pretrained_model = " résult: ⚙️，默认尺寸：512×6"
            large_pretrained_model = "🔍🌐尺寸：1024×12"

        saved_tip = "💾"

    class install_model(Locale.install_model):
        tip = '📚)>> 📂>> 💾>> 🔗>> `.sf_pkg/`.h0_ddsp_pkg_model">'
        file_label = "🧶📦 ➕ Modelo"
        model_name_label = "🤖📝"
        model_name_placeholder = "👋 输入模型名称"
        submit_btn_value = "💡🤖🔍 Modelo de Instalación"

    class path_chooser(Locale.path_chooser):
        input_path_label = "폴더"
        output_path_label = "endir 文件夹"

    class fish_audio_preprocess(Locale.fish_audio_preprocess):
        to_wav_tab = "🎶تحويل lượng كبير 🎤"
        slice_audio_tab = "👋🏼"
        preprocess_tab = "🤖📝"
        max_duration_label = "最长时间段"
        submit_btn_value = "🔄"
        input_output_same_tip = "🔗➡️"
        input_path_not_exist_tip = "🔍 Đường dẫn không tồn tại"

    class vocal_separation(Locale.vocal_separation):
        input_audio_label = "🎶🎧"
        input_path_label = "🔍 📁"
        output_path_label = "endir 👉🏼"
        use_batch_label = "🤔"
        use_de_reverb_label = "👋🏼"
        use_harmonic_remove_label = "🎶🎤"
        submit_btn_value = "🔄"
        vocal_label = "👋🏼"
        inst_label = "演奏🎵-同伴🎸"
        batch_output_message_label = "📝🤖📢💥"
        no_file_tip = ":no_file_folder_with_lock:."
        no_input_tip = "🔍폴더 선택 안 함"
        no_output_tip = "폴더를 선택하지 않았습니다"
        input_not_exist_tip = "📁🔍"
        output_not_exist_tip = "📁"
        input_output_same_tip = "📂✅命名为同一文件夹"
        finished = "🏁"
        error_when_processing = "🔍🔧🚨出现问题啦！查阅日志获取帮助📷"
        unusable_file_tip = "👋🏼 | 🎵 | 🔀 | 💾"
        batch_progress_desc = "📈"
        job_to_progress_desc = {
            "🎤🎶": "🎶🚫",
            "👋🏼": "🎶🚫",
            "🤖": "👋🏼",
            "🎤🎧": "🎶🎤",
        }

    class common_infer(Locale.common_infer):
        audio_label = "🎶🎧"
        use_batch_label = "🤔"
        use_vocal_separation_label = "🎶剔除非演奏部分"
        use_vocal_separation_info = "演奏🎵 是否要去掉背景音乐🎶？"
        use_de_reverb_label = "👋🏼"
        use_de_reverb_info = "🚫Echo️️"
        use_harmonic_remove_label = "👋🌍🎵🚫"
        use_harmonic_remove_info = ":noises_off:"
        f0_label = "🔍🤖"
        f0_info = "🎤🎧🔍🔧🤖"
        keychange_label = "👋🏼"
        keychange_info = "👩️\u200d剃鬍子🔄(man to woman) 12，👸🏼剃鬍子🔄(woman to man) -12，🗣️音色不像可以调节这个"
        threshold_label = "一刀切阈值"
        threshold_info = "👋🏻/audio_slices_threshold_for_voiced_samples, adjust to -40 or higher if there's background noise"

    class ddsp_based_infer(Locale.ddsp_based_infer):
        method_label = " kontroler"
        method_info = "👋🏼📚🔍🤖🔥"
        infer_step_label = "🔍🚶\u200d♂️"
        infer_step_info = "🔍🚶\u200d♂️ 默认就是这样"
        t_start_label = "👋"
        t_start_info = "🤔"
        num_formant_shift_key_label = "🔄📈"
        num_formant_shift_key_info = "🎵🎤📈发声音越尖锐🎵🎤📉发声音越粗糙"

    class ddsp_based_preprocess(Locale.ddsp_based_preprocess):
        method_label = "🔍🤖"
        method_info = "👋🏼📚🔍🤖🔥"

    class common_preprocess(Locale.common_preprocess):
        encoder_label = "🎶🎧🚀🤖"
        encoder_info = "🎶🔍🎵📝🤖"
        f0_label = "🔍🤖"
        f0_info = "🎤🎧🔍🔧🤖"

    class sovits(Locale.sovits):
        dataset_not_complete_tip = "🔍🚫🔄📊📈"
        finished = "🏁"

        class train_main(Locale.sovits.train_main):
            log_interval_label = "ログイン間隔"
            log_interval_info = "👋🤖.every 🕒 steps ⚡log"
            eval_interval_label = "🔍.spacing"
            eval_interval_info = "💾每隔N步保存并与验证"
            all_in_mem_label = "🔍🔄📊🌐"
            all_in_mem_info = "💡📚➡️🔍🔄🤖🧠📈-memory"
            keep_ckpts_label = "🔍📝"
            keep_ckpts_info = "留守最后的 N 度检查点"
            batch_size_label = "🏃\u200d♂️👥💨"
            batch_size_info = "🔍📈📷📝🧩"
            learning_rate_label = "🔍"
            learning_rate_info = "🔍"
            num_workers_label = "🔄📈📊"
            num_workers_info = "💻🔥🚀🔧≧４➡️⚡,+🎯🔍👍"
            half_type_label = "🔍"
            half_type_info = "🤔💥➡️)>>👌🏼✨📈%/的风险升高了，可以变得更快。"

        class train_diff(Locale.sovits.train_diff):
            batchsize_label = "🏃\u200d♂️👥💨"
            batchsize_info = "🔍✨➡️📈❗️📷ⁿ➡️💾🚫🔥👉🏼🔢"
            num_workers_label = "🏃\u200d♂️"
            num_workers_info = " 若要你的显卡不错，你可以设置为 0"
            amp_dtype_label = "🔍 Tiến độ 📊"
            amp_dtype_info = "😋🎵🔍⚡️🎮💥🔥📈⏰🚀"
            lr_label = "🔍"
            lr_info = "🚫:no_action:"
            interval_val_label = "🔍.spacing"
            interval_val_info = "🔍每隔Ν步骤检查一遍，并且储存"
            interval_log_label = "ログイン間隔"
            interval_log_info = "👋🤖.every 🕒 steps ⚡log"
            interval_force_save_label = "🔍💾🔄🕒"
            interval_force_save_info = "🔄حفظ النموذج كل N الخطوات"
            gamma_label = "👋🏼"
            gamma_info = "🚫:no_action:"
            cache_device_label = "🔍🔋🌐"
            cache_device_info = "👋🌍💻📈🔥🔍📷➡️📸🎥🎥🎥GPU++\n\n>Note: I've used '+' symbol to maintain markdown formatting and separate the output into different sentences or phrases as per the input. The 'GPU++' represents \"greater performance\" since GPUs are often associated with speed in computing."
            cache_all_data_label = "📜➡️🔍📚"
            cache_all_data_info = "🚀📈✨📝💻📊🔍🔧💥 multeramemory"
            epochs_label = "🔄(Maximum Training Rounds)"
            epochs_info = "🤖📚🔍💡🛠️🔧🔄"
            use_pretrain_label = "🔍🤖"
            use_pretrain_info = "🔄🔍⏰🛠️📚🚫"

        class train_cluster(Locale.sovits.train_cluster):
            cluster_or_index_label = "🔍📚"
            cluster_or_index_info = "🔍 Modelo de agrupamiento o de recuperación, la recuperación es ligeramente mejor que la agrupación."
            use_gpu_label = "GPU🚀💡"
            use_gpu_info = "🔄⚡️🔧🔍🔍💡📚"

        class infer(Locale.sovits.infer):
            cluster_infer_ratio_label = "📊\\/📈"
            cluster_infer_ratio_info = "📊%/📈-feature-ratio,范畴️:0-1，默认值是0，当未训练聚类模型或特征检索时。"
            linear_gradient_info = "🎶🎧 ➕ ✨🕰️"
            linear_gradient_label = "🌈-Length Adjustments"
            k_step_label = "🚶🌍"
            k_step_info = "👋🏼 | 🌍 | ➡️ | 📊 | 🔄 | 🔢 | 100"
            enhancer_adaptive_key_label = "🔄👍"
            enhancer_adaptive_key_info = "演奏者的声音能够覆盖更大的范围 | 默认值是0"
            f0_filter_threshold_label = "🔍ParameterValue"
            f0_filter_threshold_info = "👋🏻 🌍 若要在 Python 中实现一个将输入转换为仅包含表情符号的翻译程序，并保持 Markdown 格式不变，可以使用一些 API 或库来获取特定的 emoji。然而，请注意，这个任务可能非常具有挑战性且效果可能不理想，因为自然语言与表情符号的对应不是一一对应的。\n\nPython 没有内置函数能够直接将文本翻译成 emoji。一个实际应用这种方法的方法可能是编写一个字典或者使用外部 API（如通过网络请求获取某个词的 emoji）。以下是一个简单的示例使用 Python 字典来处理特定英文短语到 emoji 的翻译：\n\n```python\nemoji_dict = {\n    '你好': ':wave:',\n    '有效': ':ok:',\n    '数值范围': ':range:',\n    '降低该值': ':down_arrow:',\n    '增加哑音': ':volume_down:',\n}\n\ndef translate_to_emojis(text):\n    lines = text.split('\\n')\n    result_lines = []\n    \n    for line in lines:\n        parts = line.strip().split('.')\n        \n        # 处理每部分，如果在 emoji_dict 中找到对应项，则替换为 emoji，否则保留原样\n        translated_parts = [emoji_dict.get(part, part) for part in parts]\n        \n        result_lines.append('.'.join(translated_parts))\n    \n    return '\\n'.join(result_lines)\n\ninput_text = \"\"\"\n只有使用crepe时有效. 数值范围从0-1. \n降低该值可减少跑调概率，但会增加哑音\n\"\"\"\n\noutput_text = translate_to_emojis(input_text)\nprint(output_text)\n```\n\n这个示例代码仅使用预先定义好的 emoji 字典来替换输入文本中的部分单词。然而，请注意这只能处理特定的英语短语到emoji的映射，并且实际效果可能会很有限，尤其对于复杂的自然语言结构和非预定义词汇。\n\n如果你想更全面地实现这个功能（比如支持中文和其他语言），你可能需要使用机器翻译引擎，例如 Google Translate API 或其他类似的云服务。然而，这通常涉及到高昂的成本或需要复杂的身份验证过程，并且还可能需要处理文本切分、语法分析等自然语言处理任务来确保正确匹配单词和表情符号。\n\n请根据实际需求调整此代码以实现更完整的功能。"
            audio_predict_f0_label = "🤖🔍"
            audio_predict_f0_info = "🎤🔄🎶🎵🚫🎧"
            second_encoding_label = ""
            second_encoding_info = "👋🏽 🎤 🔇 📢 🎶 💯 🙄"
            clip_label = "📝切割长度强制执行"
            clip_info = " músicː⃣ slice ⏰ length, 0 ⚫ no enforce"

        class preprocess(Locale.sovits.preprocess):
            use_diff_label = "🏃\u200d♂️📚🔍🤖 représenterait la traduction en Emojilang de \"训练浅扩散\". Note que l'Emojilang utilise souvent des symboles plus généraux pour simuler le sens d'une phrase ou d'un terme."
            use_diff_info = (
                "🔄 若要生成训练深散播所需的档案，则需选取此项，但相对耗时较长"
            )
            vol_aug_label = "演奏🎵🎶，\n\nInput: 响度嵌入\nOutput: 🎵🎸，\n\n或者如果是指音量或声音的“响度”，可以使用：\n🔊、\n\n取决于具体语境和你想表达的意思。在Markdown中，这应该会被表示为：\n\n```\n演奏🎵🎶，\n\n或者如果是指音量或声音的“响度”，可以使用：\n🔊、\n```"
            vol_aug_info = "🎶💡🔊🔍"
            num_workers_label = " Tiến trình số"
            num_workers_info = "📚🚀"
            subprocess_num_workers_label = "🔢🧶"
            subprocess_num_workers_info = "📚🚀"
            debug_label = "💡🔍"
            debug_info = "💡📝🔧📣📢📢🚫❗"

        class model_types(Locale.sovits.model_types):
            main = " Modelo_principal"
            diff = "🔍️⁺⁻"
            cluster = "🔍📚"

        class model_chooser_extra(Locale.sovits.model_chooser_extra):
            enhance_label = "BOSE🎶🎧🔊"
            enhance_info = "🎶🎧📈🔍📢🤖🗣️📝📚🔄🔥📉🚫💡🌍"
            feature_retrieval_label = "💡🔍人脸识别提取"
            feature_retrieval_info = "🔍🤖📈🚫"

    class ddsp6(Locale.ddsp6):
        infer_tip = "🔍🤖🎧🎶"

        class model_types(Locale.ddsp6.model_types):
            cascade = "🤔"

        class train(Locale.ddsp6.train):
            batch_size_label = "🏃\u200d♂️👥💨"
            batch_size_info = "🔍✨➡️📈❗️📷ⁿ➡️💾🚫🔥👉🏼🔢"
            num_workers_label = "🏃\u200d♂️"
            num_workers_info = " 若要你的显卡不错，你可以设置为 0"
            amp_dtype_label = "🔍 Tiến độ 📊"
            amp_dtype_info = "😋🎵🔍⚡️🎮💥🔥📈⏰🚀"
            lr_label = "🔍"
            lr_info = "🚫:no_action:"
            interval_val_label = "🔍.spacing"
            interval_val_info = "🔍每隔Ν步骤检查一遍，并且储存"
            interval_log_label = "ログイン間隔"
            interval_log_info = "👋🤖.every 🕒 steps ⚡log"
            interval_force_save_label = "🔍💾🔄🕒"
            interval_force_save_info = "🔄حفظ النموذج كل N الخطوات"
            gamma_label = "👋🏼"
            gamma_info = "🚫:no_action:"
            cache_device_label = "🔍🔋🌐"
            cache_device_info = "👋🌍💻📈🔥🔍📷➡️📸🎥🎥🎥GPU++\n\n>Note: I've used '+' symbol to maintain markdown formatting and separate the output into different sentences or phrases as per the input. The 'GPU++' represents \"greater performance\" since GPUs are often associated with speed in computing."
            cache_all_data_label = "📜➡️🔍📚"
            cache_all_data_info = "🚀📈✨📝💻📊🔍🔧💥 multeramemory"
            epochs_label = "🔄(Maximum Training Rounds)"
            epochs_info = "🤖📚🔍💡🛠️🔧🔄"
            use_pretrain_label = "🔍🤖"
            use_pretrain_info = "🔄🔍⏰🛠️📚🚫"

    class reflow(Locale.reflow):
        infer_tip = "🔍🤖💡"

        class train(Locale.ddsp6.train):
            batch_size_label = "🏃\u200d♂️👥💨"
            batch_size_info = "🔍✨➡️📈❗️📷ⁿ➡️💾🚫🔥👉🏼🔢"
            num_workers_label = "🏃\u200d♂️"
            num_workers_info = " 若要你的显卡不错，你可以设置为 0"
            amp_dtype_label = "🔍 Tiến độ 📊"
            amp_dtype_info = "😋🎵🔍⚡️🎮💥🔥📈⏰🚀"
            lr_label = "🔍"
            lr_info = "🚫:no_action:"
            interval_val_label = "🔍.spacing"
            interval_val_info = "🔍每隔Ν步骤检查一遍，并且储存"
            interval_log_label = "ログイン間隔"
            interval_log_info = "👋🤖.every 🕒 steps ⚡log"
            interval_force_save_label = "🔍💾🔄🕒"
            interval_force_save_info = "🔄حفظ النموذج كل N الخطوات"
            gamma_label = "👋🏼"
            gamma_info = "🚫:no_action:"
            cache_device_label = "🔍🔋🌐"
            cache_device_info = "👋🌍💻📈🔥🔍📷➡️📸🎥🎥🎥GPU++\n\n>Note: I've used '+' symbol to maintain markdown formatting and separate the output into different sentences or phrases as per the input. The 'GPU++' represents \"greater performance\" since GPUs are often associated with speed in computing."
            cache_all_data_label = "📜➡️🔍📚"
            cache_all_data_info = "🚀📈✨📝💻📊🔍🔧💥 multeramemory"
            epochs_label = "🔄(Maximum Training Rounds)"
            epochs_info = "🤖📚🔍💡🛠️🔧🔄"
            use_pretrain_label = "🔍🤖"
            use_pretrain_info = "🔄🔍⏰🛠️📚🚫"

        class model_types(Locale.reflow.model_types):
            cascade = "🤔"

    default_spk_name = "👋🏼"
    preprocess_draw_desc = "🔍分割✅集"
    preprocess_desc = "🔄🔍📚💻📢👀"
    preprocess_finished = "📝🚀🛠️🔍🔄✅"
