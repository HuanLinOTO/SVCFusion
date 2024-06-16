from .base import Locale


class zhCNLocale(Locale):
    unknown_model_type_tip = "模型类型未知，请手动选择"

    class model_chooser(Locale.model_chooser):
        submit_btn_value = "选择模型"
        model_type_dropdown_label = "模型类型"

    class model_manager(Locale.model_manager):
        pack_btn_value = "打包模型"
        pack_result_label = "打包结果"
        packing_tip = "正在打包，请勿多次点击"
        unpackable_tip = "该模型不支持打包"

        clean_log_btn_value = "清空日志(确认不再训练再清空)"

        change_model_type_info = "#### 更改模型类型"
        change_model_type_btn_value = "确认更改"
        change_success_tip = "更改成功"
        change_fail_tip = "更改失败"

    class install_model(Locale.install_model):
        tip = """
        ## 目前仅支持上传 .sf_pkg/.h0_ddsp_pkg_model 格式的模型包
        """

        file_label = "上传模型包"

        model_name_label = "模型名称"
        model_name_placeholder = "请输入模型名称"

        submit_btn_value = "安装模型"
