1. Problem (unicode1):  Understanding Unicode (1 point)
(a) What Unicode character does chr(0) return?
\x00 是一個空的元素 但與 ""不同

(b) How does this character’s string representation (__repr__()) differ from its printed representation?
print看起來沒東西

(c) What happens when this character occurs in text? It may be helpful to play around with the 
following in your Python interpreter and see if it matches your expectations:
chr(0)
print(chr(0))
"this is a test" + chr(0) + "string"
print("this is a test" + chr(0) + "string")
不顯示在print中 但實際佔據存儲

2. Problem (unicode2):  Unicode Encodings (3 points)
(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than 
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.

- Domninent Encoding way for 98% webpage
- 大小端問題，人讀的是大端序 CPU處理的是小端序，大部分的文件 內容都是英文為主，用utf-16,32反而冗於了

(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string 
into a Unicode string. Why is this function incorrect? Provide an example of an input byte 
string that yields incorrect results.

超出ASCII之外的結果就不會對了

(c) Give a two-byte sequence that does not decode to any Unicode charact(s).
b"\xe2\x41".decode("utf-8")
1110 0010 0100 0001 \xe2\x41
unicode UTF-8 編碼方式  -> 1110xxxx 10xxxxxx 10xxxxxx , 上面這個肯定不對



Problem (train_bpe_tinystories):  BPE Training on TinyStories (2 points)
(a) Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary 
size of 10,000. Make sure to add the TinyStories <|endoftext|> special token to the 
vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How 
much time and memory did training take? What is the longest token in the vocabulary? Does 
it make sense?

longest: accomplishment, time:ard 600 sec

Resource requirements: ≤ 30 minutes (no GPUs), ≤ 30 GB RAM
(b) Profile your code. What part of the tokenizer training process takes the most time?
merge ard 240 sec, pretoeknize ard 360 sec