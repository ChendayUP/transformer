from util.tokenizer import Tokenizer
tokenizer = Tokenizer()
# 分词德语文本
german_text = "Das ist ein Beispielsatz."
german_tokens = tokenizer.tokenize_de(german_text)
print("德语分词结果:", german_tokens)

# 分词英语文本
english_text = "This is an example sentence."
english_tokens = tokenizer.tokenize_en(english_text)
print("英语分词结果:", english_tokens)