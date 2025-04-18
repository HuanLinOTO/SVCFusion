# SVCFusion

这是 SVCFusion 的核心部分开源仓库，不包括前端入口部分，因此无法直接运行，如需要使用，请下载 [闭源整合包](https://sf.dysjs.com/)。

## 为什么不开源前端部分？

前端部分是 SVCFusion 的并不重要的部分，包含了很多没多少技术含量的代码

但是开源后会导致大量的盗版（比如羽毛的 SoVITS 整合包，就遭遇了大量的倒卖的情况），甚至有些倒狗修改我的版权声明，声称是 "自己优化的版本"、"速度史诗级优化"等等，这是对我的尊重的一种侵犯，他浪费的是开发者的热爱。

因此我们选择不开源前端部分，但是这不影响想要学习的人学习真正的核心部分。

## 工具

### i18n 语言同步工具

```sh
python -m scripts.locale_sync
```

### i18n 生成基础语言类型工具

```sh
python -m scripts.spawn_base_locale
```

## 参考/引用

如有遗漏，请联系我

- [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
- [huanlinoto/so-vits-svc](https://github.com/huanlinoto/so-vits-svc) (forked and customized from [svc-develop-team/so-vits-svc](https://github.com/svc-develop-team/so-vits-svc))
- [huanlinoto/DDSP-SVC](https://github.com/huanlinoto/DDSP-SVC) (forked and customized from [yxlllc/DDSP-SVC](https://github.com/yxlllc/DDSP-SVC/)
- [huanlinoto/ReFlow-VAE-SVC](https://github.com/huanlinoto/ReFlow-VAE-SVC) (forked and customized from [yxlllc/ReFlow-VAE-SVC](https://github.com/yxlllc/ReFlow-VAE-SVC/)
