# import time
# import librosa
# import soundfile as sf
# import numpy as np
# import sounddevice as sd
# from loguru import logger
# from SVCFusion.infer_utils import infer_core

# input_devices = None
# output_devices = None
# input_devices_indices = None
# output_devices_indices = None
# stream = None

# def get_devices():
#     global input_devices, output_devices, input_devices_indices, output_devices_indices
#     sd._terminate()
#     sd._initialize()
#     devices = sd.query_devices()
#     hostapis = sd.query_hostapis()
#     for hostapi in hostapis:
#         for device_idx in hostapi["devices"]:
#             devices[device_idx]["hostapi_name"] = hostapi["name"]
#     input_devices = [
#         f"{d['name']} ({d['hostapi_name']})"
#         for d in devices
#         if d["max_input_channels"] > 0
#     ]
#     output_devices = [
#         f"{d['name']} ({d['hostapi_name']})"
#         for d in devices
#         if d["max_output_channels"] > 0
#     ]
#     input_devices_indices = [d["index"] for d in devices if d["max_input_channels"] > 0]
#     output_devices_indices = [
#         d["index"] for d in devices if d["max_output_channels"] > 0
#     ]
#     return input_devices, output_devices

# def set_devices(input_device, output_device):
#     global input_devices_indices, output_devices_indices, input_devices, output_devices
#     '''设置输出设备'''
#     sd.default.device[0] = input_devices_indices[input_devices.index(input_device)]
#     sd.default.device[1] = output_devices_indices[output_devices.index(output_device)]
#     print("input device:" + str(sd.default.device[0]) + ":" + str(input_device))
#     print("output device:" + str(sd.default.device[1]) + ":" + str(output_device))

# _keychange, _method, _infer_step, _t_start, _f0_extractor, _threhold, _spk = None, None, None, None, None, None, None

# def callback(indata: np.ndarray, outdata, frames, _time, status):
#     global _keychange, _method, _infer_step, _t_start, _f0_extractor, _threhold, _spk
#     # logger.info("callback")
#     if status:
#         print(status)
#     # outdata[:] = indata
#     # 转置 indata
#     audio = indata
#     # 启动计时器
#     start = time.time()
#     logger.info("shape: " + str(audio.shape))
#     sf.write("text.wav", audio, samplerate=44100)
#     audio, sample_rate = librosa.load("text.wav", sr=None)
#     result, sr, f0 = infer_core(
#         audio=audio,
#         sample_rate=44100,
#         keychange=_keychange,
#         method=_method,
#         infer_step=_infer_step,
#         t_start=_t_start,
#         f0_extractor=_f0_extractor,
#         threhold=_threhold,
#         cache_md5=None,
#         spk=0
#     )
#     outdata[:] = result.reshape(-1, 1)[:outdata.shape[0]]
#     print(result[0])
#     end = time.time()
#     # outdata[:] = audio
#     logger.info(f"算法延迟: {end - start}ms")

# def start_stream(input_device, output_device, chunk_size, keychange, method, infer_step, t_start, f0_extractor, threhold, spk):
#     global stream, _keychange, _method, _infer_step, _t_start, _f0_extractor, _threhold, _spk

#     _keychange = keychange
#     _method = method
#     _infer_step = infer_step
#     _t_start = t_start
#     _f0_extractor = f0_extractor
#     _threhold = threhold
#     _spk = spk

#     set_devices(input_device, output_device)
#     stream = sd.Stream(
#         channels=1,
#         callback=callback,
#         blocksize=int(chunk_size * 0.001 * 44100),
#         samplerate=44100,
#         dtype="float32")
#     logger.info("Realtime stream started.")
#     stream.start()
