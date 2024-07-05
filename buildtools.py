import os
import math
import base64
import platform
import ctypes
import binascii
import random
import shutil
import string
import subprocess
from ctypes import wintypes, windll
from win32com.client import GetObject

import tkinter as tk
from tkinter import font
from tkinter import messagebox

from package_utils.const_vars import FOUZU, WORK_DIR_PATH


class SHA256_encrypt:
    def __init__(self):
        self.constants = (
            0x428A2F98,
            0x71374491,
            0xB5C0FBCF,
            0xE9B5DBA5,
            0x3956C25B,
            0x59F111F1,
            0x923F82A4,
            0xAB1C5ED5,
            0xD807AA98,
            0x12835B01,
            0x243185BE,
            0x550C7DC3,
            0x72BE5D74,
            0x80DEB1FE,
            0x9BDC06A7,
            0xC19BF174,
            0xE49B69C1,
            0xEFBE4786,
            0x0FC19DC6,
            0x240CA1CC,
            0x2DE92C6F,
            0x4A7484AA,
            0x5CB0A9DC,
            0x76F988DA,
            0x983E5152,
            0xA831C66D,
            0xB00327C8,
            0xBF597FC7,
            0xC6E00BF3,
            0xD5A79147,
            0x06CA6351,
            0x14292967,
            0x27B70A85,
            0x2E1B2138,
            0x4D2C6DFC,
            0x53380D13,
            0x650A7354,
            0x766A0ABB,
            0x81C2C92E,
            0x92722C85,
            0xA2BFE8A1,
            0xA81A664B,
            0xC24B8B70,
            0xC76C51A3,
            0xD192E819,
            0xD6990624,
            0xF40E3585,
            0x106AA070,
            0x19A4C116,
            0x1E376C08,
            0x2748774C,
            0x34B0BCB5,
            0x391C0CB3,
            0x4ED8AA4A,
            0x5B9CCA4F,
            0x682E6FF3,
            0x748F82EE,
            0x78A5636F,
            0x84C87814,
            0x8CC70208,
            0x90BEFFFA,
            0xA4506CEB,
            0xBEF9A3F7,
            0xC67178F2,
        )
        self.h = (
            0x6A09E667,
            0xBB67AE85,
            0x3C6EF372,
            0xA54FF53A,
            0x510E527F,
            0x9B05688C,
            0x1F83D9AB,
            0x5BE0CD19,
        )

    def rightrotate(self, x, b):
        return ((x >> b) | (x << (32 - b))) & ((2**32) - 1)

    def Pad(self, W):
        return (
            bytes(W, "ascii")
            + b"\x80"
            + (b"\x00" * ((55 if (len(W) % 64) < 56 else 119) - (len(W) % 64)))
            + ((len(W) << 3).to_bytes(8, "big"))
        )

    def Compress(self, Wt, Kt, A, B, C, D, E, F, G, H):
        rightrot_E_06 = self.rightrotate(E, 6)
        rightrot_E_11 = self.rightrotate(E, 11)
        rightrot_E_25 = self.rightrotate(E, 25)
        rightrot_A_02 = self.rightrotate(A, 2)
        rightrot_A_13 = self.rightrotate(A, 13)
        rightrot_A_22 = self.rightrotate(A, 22)

        ch = (E & F) ^ (~E & G)
        temp1 = H + rightrot_E_06 ^ rightrot_E_11 ^ rightrot_E_25 + ch + Wt + Kt

        maj = (A & B) ^ (A & C) ^ (B & C)
        temp2 = rightrot_A_02 ^ rightrot_A_13 ^ rightrot_A_22 + maj

        new_H = (temp1 + temp2) & 0xFFFFFFFF
        new_D = (D + temp1) & 0xFFFFFFFF

        return new_H, A, B, C, new_D, E, F, G

    def hash(self, message):
        with open(message, "rb") as f:
            message = f.read()

        hex_message = binascii.hexlify(message).decode("utf-8")
        hex_message = self.Pad(hex_message)

        digest = list(self.h)

        for i in range(0, len(message), 64):
            S = message[i : i + 64]
            W = [int.from_bytes(S[e : e + 4], "big") for e in range(0, 64, 4)] + (
                [0] * 48
            )

            for j in range(16, 64):
                s0 = (
                    self.rightrotate(W[j - 15], 7)
                    ^ self.rightrotate(W[j - 15], 18)
                    ^ (W[j - 15] >> 3)
                )
                s1 = (
                    self.rightrotate(W[j - 2], 17)
                    ^ self.rightrotate(W[j - 2], 19)
                    ^ (W[j - 2] >> 10)
                )
                W[j] = (W[j - 16] + s0 + W[j - 7] + s1) & 0xFFFFFFFF

            A, B, C, D, E, F, G, H = digest

            for j in range(64):
                A, B, C, D, E, F, G, H = self.Compress(
                    W[j], self.constants[j], A, B, C, D, E, F, G, H
                )

            digest = [
                (x + y) & 0xFFFFFFFF for x, y in zip(digest, (A, B, C, D, E, F, G, H))
            ]

        return "".join(
            format(h, "02x") for h in b"".join(d.to_bytes(4, "big") for d in digest)
        ), "9j6gEKs5EbVI6m5Keis3"


############################################################################################################
def sub_0076(pi_digits_length=100, key_length=16):
    def get_pi_decimal(digits):
        pi_str = str(math.pi)[2:]  # 获取 π 的小数部分
        return pi_str[:digits]

    def generate_password_table(pi_decimals):
        charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        password_table = {}
        for i, digit in enumerate(pi_decimals):
            password_table[digit] = charset[i % len(charset)]
        return password_table

    def generate_key(password_table, length):
        key = "".join(password_table[d] for d in sorted(password_table.keys())[:length])
        return key

    pi_decimals = get_pi_decimal(pi_digits_length)
    password_table = generate_password_table(pi_decimals)
    key = generate_key(password_table, key_length)

    return key


def sub_0098(data, key, salt="{SALT}"):
    if isinstance(data, str):
        data = data.encode()
    if isinstance(key, str):
        key = key.encode()
    if isinstance(salt, str):
        salt = salt.encode()
        key += salt

    extended_key = key * (len(data) // len(key)) + key[: len(data) % len(key)]

    return bytes(a ^ b for a, b in zip(data, extended_key)), "EdEKlk4zMul4bwP2j5gG"


############################################################################################################
if not hasattr(wintypes, "DWORD_PTR"):
    if platform.architecture()[0] == "64bit":
        wintypes.DWORD_PTR = ctypes.c_ulonglong
    else:
        wintypes.DWORD_PTR = ctypes.c_ulong


def get_cpu_info():
    class SYSTEM_INFO(ctypes.Structure):
        _fields_ = [
            ("wProcessorArchitecture", wintypes.WORD),
            ("wReserved", wintypes.WORD),
            ("dwPageSize", wintypes.DWORD),
            ("lpMinimumApplicationAddress", wintypes.LPVOID),
            ("lpMaximumApplicationAddress", wintypes.LPVOID),
            ("dwActiveProcessorMask", wintypes.DWORD_PTR),
            ("dwNumberOfProcessors", wintypes.DWORD),
            ("dwProcessorType", wintypes.DWORD),
            ("dwAllocationGranularity", wintypes.DWORD),
            ("wProcessorLevel", wintypes.WORD),
            ("wProcessorRevision", wintypes.WORD),
        ]

    system_info = SYSTEM_INFO()
    windll.kernel32.GetSystemInfo(ctypes.byref(system_info))
    return (
        system_info.wProcessorArchitecture,
        system_info.dwProcessorType,
        "99BM5avrAdzxeyJEa2Xz",
    )


def get_bios_info():
    objWMI = GetObject("winmgmts:root\\cimv2")
    bios = objWMI.ExecQuery("SELECT * FROM Win32_BIOS")
    for item in bios:
        return item.SMBIOSBIOSVersion, "ZVCcNY7YqSVZrniVJZDh"


def get_windows_serial_number():
    objWMI = GetObject("winmgmts:root\\cimv2")
    os = objWMI.ExecQuery("SELECT * FROM Win32_OperatingSystem")
    for item in os:
        return item.SerialNumber, "yAWFon6ZrwJLEmmGEv5r"


def get_drive_serial_number(drive):
    volume_serial_number = wintypes.DWORD()
    file_system_flags = wintypes.DWORD()
    file_system_name_buffer = ctypes.create_unicode_buffer(261)
    volume_name_buffer = ctypes.create_unicode_buffer(261)
    ctypes.windll.kernel32.GetVolumeInformationW(
        ctypes.c_wchar_p(drive),
        volume_name_buffer,
        ctypes.sizeof(volume_name_buffer),
        ctypes.byref(volume_serial_number),
        None,
        ctypes.byref(file_system_flags),
        file_system_name_buffer,
        ctypes.sizeof(file_system_name_buffer),
    )
    return volume_serial_number.value, "DJkgxCZ42Z4xdRIW7oIy"


def get_all_drive_serial_numbers():
    drive_serials = {}
    bitmask = ctypes.windll.kernel32.GetLogicalDrives()
    for i in range(26):
        if bitmask & (1 << i):
            drive_letter = f"{chr(65 + i)}:\\"
            serial_number, magic = get_drive_serial_number(drive_letter)
            if magic != "DJkgxCZ42Z4xdRIW7oIy":
                exit(1)
            drive_serials[drive_letter] = serial_number
    return drive_serials, "EdEKlk4zMul4bwP2j5gG"


def get_memory_serial_numbers():
    objWMI = GetObject("winmgmts:root\\cimv2")
    memory_modules = objWMI.ExecQuery("SELECT * FROM Win32_PhysicalMemory")
    serial_numbers = [module.SerialNumber.strip() for module in memory_modules]
    return serial_numbers, "aXnewWmJnKnjhi0FV85Q"


############################################################################################################
def on_agree(current_system_serial_base64):
    file_path = "dist.cp310-win_amd64.pyd"

    key = os.urandom(16)
    SHA256_encoder = SHA256_encrypt()
    webui_sha256, magic = SHA256_encoder.hash(file_path)
    if magic != "9j6gEKs5EbVI6m5Keis3":
        exit(1)
    license_bin_1, magic = sub_0098(webui_sha256, key)
    if magic != "EdEKlk4zMul4bwP2j5gG":
        exit(1)
    license_bin_2, magic = sub_0098(current_system_serial_base64, key)
    if magic != "EdEKlk4zMul4bwP2j5gG":
        exit(1)
    license_bin = license_bin_1 + license_bin_2

    with open(
        os.path.join(os.path.dirname("dist.cp310-win_amd64.pyd"), "license.bin"), "wb"
    ) as f:
        f.write(key + license_bin)

    messagebox.showinfo("信息", "感谢使用 SVC Fusion 整合包！")
    # subprocess.Popen(["./.conda/python.exe", "launcher.py"])
    root.destroy()
    TEST_UI(salt="{SALT}")


def on_disagree():
    root.destroy()


def open_main_url(*args):
    import webbrowser

    webbrowser.open("https://sf.dysjs.com/")


def open_faq_url(*args):
    import webbrowser

    webbrowser.open("https://sf.dysjs.com/faq")


def update_countdown():
    global countdown
    if countdown > 0:
        agree_button.config(text=f"同意 ({countdown})")
        countdown -= 1
        root.after(1000, update_countdown)
    else:
        agree_button.config(text="同意", state=tk.NORMAL)


def ui(current_system_serial_base64):
    global root
    global agree_button
    global countdown
    root = tk.Tk()

    root.title("SVC Fusion用户协议")

    custom_font = font.Font(family="DengXian", size=16, weight="bold")

    agreement_text = """
    本整合包完全免费，禁止任何形式的售卖，包括但不限于收费教学、VIP课程等！！！
    如果你是付费获取的，请立即退款！！！

    本整合包完全免费，禁止任何形式的售卖，包括但不限于收费教学、VIP课程等！！！
    如果你是付费获取的，请立即退款！！！

    禁止用于商业用途，禁止用于违法活动，否则后果自负！！！

    整合包唯一官方网址：https://sf.dysjs.com/

    整合包常见报错网址：https://sf.dysjs.com/faq
    """

    text_widget = tk.Text(root, font=custom_font, wrap="word", height=13, width=100)
    text_widget.insert(tk.END, agreement_text)

    # Add a tag to the URL to make it clickable
    text_widget.tag_add("url", "10.14", "10.39")
    text_widget.tag_config("url", foreground="blue", underline=True)
    text_widget.tag_bind("url", "<Button-1>", open_main_url)

    text_widget.tag_add("url", "12.14", "12.42")
    text_widget.tag_config("url", foreground="blue", underline=True)
    text_widget.tag_bind("url", "<Button-1>", open_faq_url)

    text_widget.config(state=tk.DISABLED)
    text_widget.pack(pady=10, padx=20)

    countdown = 2
    agree_button = tk.Button(
        root,
        text=f"同意 ({countdown})",
        font=custom_font,
        state=tk.DISABLED,
        command=lambda: on_agree(current_system_serial_base64),
    )
    agree_button.pack(side=tk.RIGHT, padx=20, pady=20)

    tk.Button(root, text="不同意", font=custom_font, command=on_disagree).pack(
        side=tk.LEFT, padx=20, pady=20
    )

    root.after(1000, update_countdown)

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    window_width = 900
    window_height = 400

    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)

    root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

    root.resizable(False, False)

    root.mainloop()


def main():
    cpu_architecture, cpu_type, magic = get_cpu_info()
    if magic != "99BM5avrAdzxeyJEa2Xz":
        exit(1)
    bios_info, magic = get_bios_info()
    if magic != "ZVCcNY7YqSVZrniVJZDh":
        exit(1)
    windows_serial_number, magic = get_windows_serial_number()
    if magic != "yAWFon6ZrwJLEmmGEv5r":
        exit(1)
    drive_serial_numbers, magic = get_all_drive_serial_numbers()
    if magic != "EdEKlk4zMul4bwP2j5gG":
        exit(1)
    memory_serial_numbers, magic = get_memory_serial_numbers()
    if magic != "aXnewWmJnKnjhi0FV85Q":
        exit(1)

    identifiers = (
        [str(cpu_type), bios_info, windows_serial_number, str(cpu_architecture)]
        + memory_serial_numbers
        + [str(sn) for sn in drive_serial_numbers.values()]
    )

    current_system_serial = "".join(identifiers)
    current_system_serial_base64 = base64.b64encode(
        current_system_serial.encode()
    ).decode()

    license_path = os.path.join(
        os.path.dirname("dist.cp310-win_amd64.pyd"), "license.bin"
    )
    if os.path.exists(license_path):
        with open(license_path, "rb") as f:
            stored_key = f.read(16)
            stored_webui_sha256 = f.read(64)
            stored_system_serial_base64 = f.read()

            decrypted_webui_sha256, magic = sub_0098(stored_webui_sha256, stored_key)
            if magic != "EdEKlk4zMul4bwP2j5gG":
                exit(1)
            decrypted_base64_encoded_info, magic = sub_0098(
                stored_system_serial_base64, stored_key
            )
            if magic != "EdEKlk4zMul4bwP2j5gG":
                exit(1)

            current_hash, magic = SHA256_encrypt().hash("dist.cp310-win_amd64.pyd")
            if magic != "9j6gEKs5EbVI6m5Keis3":
                exit(1)

            if (
                current_hash == decrypted_webui_sha256.decode()
                and current_system_serial_base64
                == decrypted_base64_encoded_info.decode()
            ):
                print("License check passed.")
                TEST_UI(salt="{SALT}")  # 在这里插入webui的入口点
            else:
                print("License file broken.")
                ui(current_system_serial_base64)
    else:
        print("License file not found.")
        ui(current_system_serial_base64)


def entry_point():
    main()


def TEST_UI(salt=None):
    encrypted_function_code = b"{ENTRY}"
    key = sub_0076()
    decrypted_function, magic = sub_0098(encrypted_function_code, key, salt)
    # 使用一个局部字典来执行并保存局部变量
    local_scope = {}
    exec(decrypted_function.decode(), globals(), local_scope)
    local_scope["tobeornottobe_isaquestion"]({NUMBERS})
    demo.launch(inbrowser=True)


# del start
def generate_random_function():
    # 定义参数名称
    param_names = ["a", "b", "c", "d", "e"]

    # 生成随机表达式的函数
    def generate_expression(depth=3):
        if depth == 0:
            # 终止条件，返回一个参数或一个常数
            return random.choice(param_names + [str(random.randint(1, 10))])

        # 选择一个随机操作符
        operator = random.choice(["+", "-", "*", "/"])

        # 递归生成左右子表达式
        left = generate_expression(depth - 1)
        right = generate_expression(depth - 1)

        # 确保除法不出现除以零的情况
        if operator == "/" and right == "0":
            right = "1"

        return f"({left} {operator} {right})"

    # 生成一个随机表达式
    expression = generate_expression()

    # 创建函数字符串
    func_str = f"""
def tobeornottobe_isaquestion(a, b, c, d, e):
    return {expression}
"""
    return func_str


def encrypt_function(salt):
    # 要加密的函数
    # function_code
    # 读取 webui.py 到 function_code
    # with open("webui.py", "r", encoding="utf-8") as f:
    #     function_code = str(f.read())
    function_code = generate_random_function()
    #     function_code = """
    # from package_utils.const_vars import WORK_DIR_PATH
    # def launch():
    #     print(WORK_DIR_PATH)
    # """

    key = sub_0076()  # 生成密钥

    encrypted_data, magic = sub_0098(function_code, key, salt=salt)  # 加密函数
    return encrypted_data


def replace_between(string, a, b, replacement):
    start_index = string.find(a)
    if start_index == -1:
        return string

    end_index = string.find(b, start_index + len(a))
    if end_index == -1:
        return string

    end_index += len(b)
    print(f"Replacing between {start_index} and {end_index}, {replacement}")
    return string[:start_index] + replacement + string[end_index:]


def random_n_char(n=6):
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


if __name__ == "__main__":
    salt = random_n_char()
    encrypted_data = encrypt_function(salt)
    # 读取 webui.py
    with open("webui.py", "r", encoding="utf-8") as f:
        webui = str(f.read())
    dist = webui
    dist = replace_between(dist, "# del" + " start", "# del" + " end", "")

    with open("buildtools.py", "r", encoding="utf-8") as f:
        dist += "\n" + str(f.read())
    dist = replace_between(dist, "# del" + " start", "# del" + " end", "")
    dist = dist.replace('b"{ENTRY}"', str(encrypted_data))
    dist = dist.replace("{SALT}", str(salt))
    dist = dist.replace(
        "{NUMBERS}",
        f"{random.randint(1, 10)},{random.randint(1, 10)},{random.randint(1, 10)},{random.randint(1, 10)},{random.randint(1, 10)}",
    )
    dist += """
def launch_dialog():
    entry_point()
"""
    # 写入 dist.py
    with open("dist.py", "w", encoding="utf-8") as f:
        f.write(dist)
# del end
