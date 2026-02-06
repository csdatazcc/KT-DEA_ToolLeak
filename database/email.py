import pandas as pd
import random
import re

# 设置随机种子
random.seed(42)

# 读取原始CSV
df = pd.read_csv("train.csv")

# 自动识别邮件正文列

# 随机抽样200条
sampled_df = df.sample(n=min(200, len(df)), random_state=42)

# 邮件拆分函数
def split_email(email_text):
    lines = email_text.splitlines()
    subject = ""
    to_field = ""
    from_field = ""
    from_lines = []
    to = ""
    # 1. 提取Subject
    for line in lines:
        if line.startswith("Subject:"):
            subject = line
            break

    # 2. 提取收件人，只取Hi开头的那一行
    for line in lines[:5]:  # 前5行通常是称呼
        if line.startswith("Hi "):
            to =line
            to_field = line.replace("Hi ", "").strip()
            break

    # 3. 提取发件人，只取Best开头下一行
    from_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Best"):
            from_start = i + 1  # 从下一行开始
            break
    if from_start is not None:
        from_lines = [line.strip() for line in lines[from_start:] if line.strip()]
    from_field = "\n".join(from_lines)

    exclude_set = set([subject, to] + from_lines)
    filtered_lines = [line for line in lines if line.strip() not in exclude_set]

    # 找第一行非空到最后一行非空
    first_non_empty_idx = None
    last_non_empty_idx = None
    for i, line in enumerate(filtered_lines):
        if line.strip():
            if first_non_empty_idx is None:
                first_non_empty_idx = i
            last_non_empty_idx = i

    if first_non_empty_idx is not None and last_non_empty_idx is not None:
        context_lines = filtered_lines[first_non_empty_idx:last_non_empty_idx+1]
    else:
        context_lines = []

    context = "\n".join(context_lines).strip()

    return pd.Series([subject, from_field, to_field, context])

# 应用拆分
split_df = sampled_df["0"].apply(split_email)
split_df.columns = ['subject', 'from', 'to', 'context']

# 保存CSV
split_df.to_csv("marketing_emails_200.csv", index=False, encoding='utf-8')
print("处理后的CSV已保存：sampled_emails.csv")
