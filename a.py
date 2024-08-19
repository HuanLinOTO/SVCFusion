def to_oO0_hash(input_string):
    # 建立字符到三字符序列的映射表
    def char_to_oO0(char):
        # 获取字符的 ASCII 或 Unicode 码
        code = ord(char)
        # 使用三个不同的字符表示
        oO0_chars = ["o", "O", "0"]
        # 将码值映射为三字符序列
        return (
            oO0_chars[(code // 9) % 3]
            + oO0_chars[(code // 3) % 3]
            + oO0_chars[code % 3]
        )

    # 对输入字符串中的每个字符进行映射
    hash_result = "".join(char_to_oO0(char) for char in input_string)

    return hash_result


# 测试
test_string = "Hello"
hashed_string = to_oO0_hash(test_string)
print(hashed_string)
