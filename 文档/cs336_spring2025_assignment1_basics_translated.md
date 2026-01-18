# CS336 作业 1 (基础): 构建 Transformer LM

版本 1.0.6

CS336 员工

2025 年春季

# 1 作业概述

在此次作业中，您将从头开始构建训练标准 Transformer 语言模型 (LM) 所需的所有组件，并训练一些模型。

# 您将实现的

1. 字节对编码 (BPE) 分词器 $(\S 2)$
2. Transformer 语言模型 (LM) (§3)
3. 交叉熵损失函数和 AdamW 优化器 (§4)
4. 训练循环，支持序列化和加载模型及优化器状态 (§5)

# 您将运行的

1. 在 TinyStories 数据集上训练一个 BPE 分词器。
2. 在数据集上运行您训练好的分词器，将其转换为整数 ID 序列。
3. 在 TinyStories 数据集上训练一个 Transformer LM。
4. 使用训练好的 Transformer LM 生成样本并评估困惑度。
5. 在 OpenWebText 上训练模型，并将您达到的困惑度提交到排行榜。

你可以使用 我们希望你从头开始构建这些组件。特别是，除了以下内容外，你不能使用 torch(nn, torch(nn)).functional 或 torch_optim 中的任何定义：

- torch(nn_PARAMETER
- torch(nn) 中的容器类（例如，Module、ModuleList、Sequential 等）<sup>1</sup>
- torch.optim.Optimizer 基类

你可以使用任何其他 PyTorch 定义。如果你想使用某个函数或类但不确定是否允许，请随时在 Slack 上提问。如有疑问，请考虑使用它是否会损害本次作业的“从头开始”精神。

关于 AI 工具的声明 允许使用 ChatGPT 等语言模型来回答低级编程问题或关于语言模型的高级概念性问题，但禁止直接使用它来解决问题。

我们强烈建议您在完成作业时禁用IDE中的AI自动补全功能（例如，Cursor Tab、GitHub Copilot）（尽管非AI自动补全，例如函数名自动补全是完全可以的）。我们发现AI自动补全功能会大大增加深入理解内容的难度。

代码示例：所有作业代码以及此文档都可以在GitHub上找到：

github.com/stanford-cs336/assignment1-basics

请git clone该仓库。如果有任何更新，我们会通知您，以便您git pull以获取最新版本。

1. cs336 Basics/*：这是您编写代码的地方。请注意，这里没有代码——您可以从头开始做任何您想做的事情！  
2. adapters.py: 您的代码必须具备一组功能。对于每一项功能（例如，缩放点积注意力），只需调用您的代码即可完成其实现（例如，runScaled.dot_productattention）。注意：您对 adapters.py 的修改不应包含任何实质性逻辑；这是胶水代码。
3. test_*.py: 这包含了您必须通过的所有测试（例如，testScaled.dot_productattention），这些测试将调用在 adapters.py 中定义的钩子。请勿编辑测试文件。

如何提交 您将把以下文件提交到 Gradescope：

- writeup.pdf：回答所有书面问题。请排版您的答案。
- code.zip：包含您编写的所有代码。

要提交到排行榜，请提交一个 PR 到：

github.com/stanford-cs336/assignment1-basics-leaderboard

有关详细的提交说明，请参阅排行榜存储库中的 README.md。

在哪里获取数据集 本次作业将使用两个预处理过的数据集：TinyStories [Eldan and Li, 2023] 和 OpenWebText [Gokaslan et al., 2019]。这两个数据集都是单个的大型纯文本文件。如果你是和班级一起做作业，可以在任何非头节点机器的 /data 目录下找到这些文件。

如果你是在家跟着做，可以使用 README.md 中的命令下载这些文件。

# 低资源/缩小规模提示：初始化

在整个课程的作业讲义中，我们将提供关于如何用更少或没有 GPU 资源来完成作业部分的建议。例如，我们有时会建议缩小数据集或模型的大小，或者解释如何在 MacOS 集成 GPU 或 CPU 上运行训练代码。你会发现这些“低资源提示”在一个蓝色的框里（像这个）。即使你是斯坦福大学的学生，可以使用课程机器，这些提示也可能帮助你更快地迭代和节省时间，所以我们建议你阅读它们！

# 低资源/降级技巧：在 Apple Silicon 或 CPU 上进行分配 1

使用员工解决方案代码，我们可以在配备 36 GB RAM 的 Apple M3 Max 芯片上训练一个 LM，在 Metal GPU (MPS) 上不到 5 分钟即可生成相当流畅的文本，使用 CPU 大约需要 30 分钟。如果这些词对您来说意义不大，请不要担心！您只需要知道，如果您有一台相对较新的笔记本电脑，并且您的实现是正确且高效的，那么您将能够训练一个小型 LM，生成具有不错流畅度的简单儿童故事。

稍后在分配中，我们将解释如果您使用的是 CPU 或 MPS，需要进行哪些更改。

# 2 字节对编码 (BPE) 分词器

<output>
在本次作业的第一部分，我们将训练并实现一个字节级字节对编码（BPE）分词器 [Sennrich et al., 2016, Wang et al., 2019]。具体来说，我们将任意（Unicode）字符串表示为字节序列，并在该字节序列上训练我们的 BPE 分词器。之后，我们将使用此分词器将文本（字符串）编码为词元（整数序列）以用于语言建模。

# 2.1 Unicode 标准
</output>

Unicode 是一个文本编码标准，它将字符映射到整数代码点。截至 Unicode 16.0（于 2024 年 9 月发布），该标准定义了 168 种书写系统中的 154,998 个字符。例如，字符 "s" 的代码点是 115（通常表示为 U+0073，其中 U+ 是约定前缀，0073 是 115 的十六进制表示），字符 "牛" 的代码点是 29275。在 Python 中，您可以使用 ord() 函数将单个 Unicode 字符转换为其整数表示。chr() 函数将整数 Unicode 代码点转换为具有相应字符的字符串。

```txt
>>>ord('牛') 29275   
>>>chr(29275) '牛'
```

# 问题 (unicode1)：理解 Unicode（1 分）

(a) chr(0) 返回哪个 Unicode 字符？

交付物：一句话回答。

(b) 该字符的字符串表示形式（__repr__()）与其打印表示形式有何不同？

交付物：一句话回答。

(c) 当此字符出现在文本中时会发生什么？在 Python 解释器中尝试以下操作可能会有所帮助，看看它是否符合您的预期：

```txt
>>>chr(0)   
>>>print(chr(0))   
>>> "this is a test"  $^+$  chr(0)  $^+$  "string"   
>>> print("this is a test"  $^+$  chr(0)  $^+$  "string")
```

交付成果：一句话回复。

# 2.2 Unicode 编码

虽然 Unicode 标准定义了从字符到码点（整数）的映射，但直接在 Unicode 码点上训练分词器是不切实际的，因为词汇表会过大（约 150K 个条目）且稀疏（因为许多字符非常罕见）。相反，我们将使用一种 Unicode 编码，它将 Unicode 字符转换为字节序列。Unicode 标准本身定义了三种编码：UTF-8、UTF-16 和 UTF-32，其中 UTF-8 是互联网上的主要编码（占所有网页的 $98\%$ 以上）。

要将Unicode字符串编码为UTF-8，我们可以使用Python中的encode()函数。要访问Python bytes对象的底层字节值，我们可以对其进行迭代（例如，调用list()）。最后，我们可以使用decode()函数将UTF-8字节字符串解码为Unicode字符串。

```txt
>>> test_string = "hello! 你怎么是"
>>> utf8-encoded = test_string.encode("utf-8")
>>> print utf8-encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!"
>>> print(typeutf8-encoded))
<class 'bytes">
>>> # 获取编码字符串的字节值（0到255的整数）。
>>> list utf8-encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
>>> # 一个字节不一定对应一个Unicode字符！
>>> print(len(test_string))
13
>>> print(len utf8-encoded))
23
>>> print utf8-encodeddecode("utf-8"))
hello! 你怎么是
```

通过将我们的 Unicode 码点转换为一系列字节（例如，通过 UTF-8 编码），我们实际上是将一系列码点（0 到 154,997 范围内的整数）转换为一系列字节值（0 到 255 范围内的整数）。长度为 256 的字节词汇表更容易处理。使用字节级分词器时，我们无需担心词汇表外词元，因为我们知道任何输入文本都可以表示为 0 到 255 的整数序列。

# 问题 (unicode2)：Unicode 编码（3 分）

(a) 与 UTF-16 或 UTF-32 相比，有哪些理由更倾向于在 UTF-8 编码的字节上训练我们的分词器？比较这些编码对各种输入字符串的输出可能有所帮助。

交付成果：一到两句话的回答。

(b) 考虑以下（不正确的）函数，它旨在将 UTF-8 字节字符串解码为 Unicode 字符串。为什么这个函数不正确？提供一个产生错误结果的输入字节字符串示例。

```python
defdecodeutf8_bytes_to_str WRONG(bytes): return".join([bytes([b]).decode("utf-8")for b in bytestring])   
>>>decodeutf8_bytes_to_str WRONG("hello".encode("utf-8")) 'hello'
```

交付物：一个示例输入字节字符串，对于该字符串 `decodeutf8_bytes_to_str WRONG` 会产生不正确的输出，并附带一句解释该函数为何不正确的说明。

(c) 给出一个不能解码为任何 Unicode 字符的两个字节序列。

交付物：一个示例，附带一句解释。

# 2.3 子词分词

虽然字节级分词可以缓解词级分词器面临的词汇表外问题，但将文本分词为字节会导致输入序列非常长。这会减慢模型训练速度，因为

<output>
一个包含10个单词的句子在词级别语言模型中可能只有10个词元，但在字符级别模型中可能包含50个或更多词元（取决于单词的长度）。处理这些更长的序列需要模型在每个步骤中进行更多的计算。此外，基于字节序列的语言建模很困难，因为更长的输入序列会在数据中产生长期依赖。

子词分词器是词级别分词器和字节级别分词器之间的一种折衷。请注意，字节级别分词器的词汇表有256个条目（字节值为0到225）。子词分词器通过更大的词汇表大小来换取对输入字节序列更好的压缩。例如，如果字节序列 b'the' 在我们的原始文本训练数据中经常出现，那么在词汇表中为其分配一个条目可以将这个3词元的序列减少到一个词元。
</output>

我们如何选择这些子词单元添加到我们的词汇表中？Sennrich 等人 [2016] 提出使用字节对编码 (BPE；Gage, 1994)，这是一种压缩算法，它迭代地将最常见的字节对替换（“合并”）为单个、新的未使用索引。请注意，此算法将子词词元添加到我们的词汇表中，以最大化输入序列的压缩——如果一个词在我们的输入文本中出现的次数足够多，它将被表示为单个子词单元。

通过 BPE 构建的词汇表的子词分词器通常称为 BPE 分词器。在此次作业中，我们将实现一个字节级 BPE 分词器，其中词汇表项是字节或字节的合并序列，这使我们在处理词汇表外单词和管理输入序列长度方面都能获得最佳效果。构建 BPE 分词器词汇表的过程称为“训练”BPE 分词器。

# 2.4 BPE 分词器训练

BPE 分词器的训练过程包含三个主要步骤。

词汇表初始化 分词器词汇表是将字节串词元映射到整数 ID 的一对一映射。由于我们训练的是字节级 BPE 分词器，因此我们的初始词汇表就是所有字节的集合。由于有 256 个可能的字节值，我们的初始词汇表大小为 256。

预分词 一旦有了词汇表，原则上，你可以计算文本中字节相邻出现的频率，并从最频繁的字节对开始合并。然而，这在计算上相当昂贵，因为每次合并都需要对语料库进行一次完整的遍历。此外，直接在语料库中合并字节可能会导致词元仅在标点符号上有所不同（例如，dog! vs. dog.）。这些词元将获得完全不同的词元 ID，尽管它们可能具有很高的语义相似性（因为它们仅在标点符号上有所不同）。

为了避免这种情况，我们对语料库进行预分词。你可以将其视为语料库上的粗粒度分词，这有助于我们计算字符对出现的频率。例如，“text”这个词可能是一个预分词，出现了10次。在这种情况下，当我们计算字符“t”和“e”相邻出现的频率时，我们会发现“text”这个词中的“t”和“e”是相邻的，我们可以将它们的计数增加10，而不是遍历整个语料库。由于我们正在训练一个字节级别的BPE模型，每个预分词都表示为UTF-8字节序列。

Sennrich 等人 [2016] 的原始BPE实现通过简单地按空格分割（即 s.split("\\")）来进行预分词。相比之下，我们将使用一个基于正则表达式的预分词器（GPT-2；Radford et al., 2019 使用的），来自 github.com/openai/tiktoken/pull/234/files：

```python
>>>PAT  $\equiv$  r""""?:[sdmt]ll|ve|re)！p{L}+|？p{N}+|？[^s\\p{L}\\p{N}]  $+\mid \backslash s + (\text{？！}\backslash S)\mid \backslash s + "''"$
```

为了更好地了解其行为，使用此预分词器（pre-tokenizer）交互式地分割一些文本可能会很有用：

>>> # 需要 `regex` 包
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")

['some', 'text', 'that', 'i', "'ll", 'pre', '-', 'tokenizer']

然而，在代码中使用时，您应该使用 re.findall 来避免在构建预词元（pre-tokens）到其计数的映射时存储预分词的词语。

计算 BPE 合并 现在我们已经将输入文本转换为预分词，并将每个预分词表示为 UTF-8 字节序列，我们可以计算 BPE 合并（即训练 BPE 分词器）。总的来说，BPE 算法会迭代地计算每对字节的出现次数，并识别出频率最高的一对（“A”，“B”）。然后，将这对最频繁的字节（“A”，“B”）的每次出现进行合并，即替换为一个新的词元“AB”。这个新的合并词元被添加到我们的词汇表中；因此，BPE 训练后的最终词汇表的大小是初始词汇表的大小（在本例中为 256），加上训练过程中执行的 BPE 合并操作的数量。为了在 BPE 训练期间提高效率，我们不考虑跨越预分词边界的字节对。在计算合并时，通过优先选择字典序更大的字节对来确定性地打破字节对频率的平局。例如，如果字节对（“A”，“B”），（“A”，“C”），（“B”，“ZZ”），和（“BA”，“A”）的频率最高，我们d merge ("BA", "A"):

```html
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")]('BA', 'A')
```

特殊标记 (Special tokens) 有时，一些字符串（例如 <|endoftext|>）被用来编码元数据（例如文档之间的边界）。在编码文本时，通常希望将某些字符串视为“特殊标记”，这些标记永远不会被拆分成多个标记（即，将始终保留为单个标记）。例如，序列结束字符串 <|endoftext|> 应始终保留为单个标记（即单个整数 ID），以便我们知道何时停止从语言模型生成。这些特殊标记必须添加到词汇表中，以便它们具有相应的固定标记 ID。

Sennrich et al. [2016] 的算法 1 包含一个低效的 BPE 分词器训练实现（基本上遵循我们上面概述的步骤）。作为第一个练习，实现并测试此函数以检验您的理解可能很有用。

# 示例 (bpe_example): BPE 训练示例

这是 Sennrich et al. [2016] 中的一个风格化示例。考虑一个包含以下文本的语料库：

low low low low low lower lower widest widest widest newest newest newest newest newest newest

并且词汇表有一个特殊词元 $ <\text{endoftext}> $。

词汇表 我们用特殊词元 $ <\text{endoftext}| $ 和 256 个字节值来初始化我们的词汇表。

预分词 为了简单起见，并专注于合并过程，在本示例中我们假设预分词只是按空格分割。当我们进行预分词和计数时，我们会得到频率表。

{low: 5, lower: 2, widest: 3, newest: 6}

将其表示为 dict[tuple[bytes], int] 会很方便，例如 $\{(1,0,w): 5\ldots\}$。请注意，即使是单个字节在 Python 中也是一个字节对象。Python 中没有字节类型来表示单个字节，就像 Python 中没有字符类型来表示单个字符一样。

我们首先查看每对连续的字节，并对它们出现的单词的频率求和 $\{10:7,0w:7,we:8,er:2,wi:3,id:3,de:3,es:9,st:9,ne:6,ew:6\}$。对 ('es') 和 ('st') 并列，所以我们取字典序更大的对 ('st')。然后我们将预词元合并，直到我们得到 $\{(1,o,w):5,(1,o,w,e,r):2,(w,i,d,e,st):3,(n,e,w,e,st):6\}$。

在第二轮中，我们看到 (e, st) 是最常见的对（计数为 9），我们将合并为 $\{(1,o,w):5,(1,o,w,e,r):2,(w,i,d,\text{est}):3,(n,e,w,\text{est}):6\}$。继续这个过程，我们最终得到的合并序列将是 ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lower r']。

如果我们进行 6 次合并，我们将得到 ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']，我们的词汇表元素将是 $\left[ < \text{endoftext} | > \right]$，[...256 BYTE CHARS]，st，est，ow，low，west，ne]。

使用这个词汇表和合并集，单词 newest 将被分词为 [ne, west]。

# 2.5 实验 BPE 分词器训练

让我们在 TinyStories 数据集上训练一个字节级 BPE 分词器。查找/下载数据集的说明可以在第 1 节中找到。开始之前，我们建议您查看 TinyStories 数据集，以了解数据的内容。

并行化预分词 您会发现一个主要的瓶颈是预分词步骤。您可以通过使用内置库 multiprocessing 来并行化代码，从而加速预分词。具体来说，我们建议在预分词的并行实现中，将语料库分块，同时确保您的块边界出现在特殊标记的开头。您可以直接使用以下链接中的入门代码来获取块边界，然后使用这些边界将工作分发到您的进程中：

https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336 Basics/pretokenization_example.py

这种分块将始终有效，因为我们从不想跨文档边界进行合并。就本次作业而言，您可以始终这样拆分。不要担心接收到一个不包含 $\langle | \text{endoftext} | \rangle$ 的非常大的语料库的边缘情况。

在预分词前移除特殊标记 在使用正则表达式模式（使用 re.finditer）运行预分词之前，您应该从语料库（或并行实现中的分块）中剥离所有特殊标记。确保您在特殊标记处进行拆分，以免在它们分隔的文本之间发生合并。例如，如果您有一个像 [Doc 1]<|endoftext|>[Doc 2] 这样的语料库（或分块），您应该在特殊标记 <|endoftext|> 处进行拆分，并分别对 [Doc 1] 和 [Doc 2] 进行预分词，以免在文档边界处发生合并。这可以通过使用 re.split 来完成，其中 "|\"].join(special_tokens) 作为分隔符（小心使用 re.escape，因为 | 可能会出现在特殊标记中）。测试 test_train_bpe_special_tokens 将对此进行测试。

优化合并步骤

上面示例中 BPE 训练的朴素实现速度很慢，因为每次合并都需要遍历所有字节对来识别最频繁的字节对。然而，每次合并后唯一改变的字节对计数是那些与合并后的字节对重叠的计数。因此，通过索引所有字节对的计数并增量更新这些计数，而不是显式地遍历每个字节对来计算字节对频率，可以提高 BPE 训练速度。通过这种缓存过程可以显著加快速度，尽管我们注意到 BPE 训练的合并部分在 Python 中是无法并行化的。

# 低资源/缩小规模技巧：分析

您应该使用 cProfile 或 scalene 等分析工具来识别实现中的瓶颈，并专注于优化它们。

# 低资源/缩小规模技巧：“缩小规模”

与其在 TinyStories 数据集上训练分词器，我们建议你先在数据的一个小子集上进行训练：“调试数据集”。例如，你可以改用 TinyStories 验证集来训练分词器，该验证集包含 22K 个文档，而不是 2.12M 个。这说明了一种通用的策略，即尽可能缩小规模以加快开发速度：例如，使用更小的数据集、更小的模型尺寸等。选择调试数据集的大小或超参数配置需要仔细考虑：你希望调试集足够大，能够反映完整配置中的瓶颈（以便你进行的优化能够泛化），但又不能太大以至于运行时间过长。

# 问题（train_bpe）：BPE 分词器训练（15 分）

交付成果：编写一个函数，该函数接收输入文本文件的路径，并训练一个（字节级）BPE 分词器。你的 BPE 训练函数应至少处理以下输入参数：

```markdown
 input_path: str 训练 BPE 分词器数据的文本文件路径。

vocab_size: int 一个正整数，用于定义最终词汇表的总大小（包括初始字节词汇表、通过合并产生的词汇项以及任何特殊标记）。

special_tokens: list[str] 要添加到词汇表中的字符串列表。这些特殊标记不会以其他方式影响 BPE 训练。

您的 BPE 训练函数应返回生成的词汇表和合并：

vocabulary: dict[int, bytes] 分词器词汇表，一个从 int（词汇表中的词元 ID）到 bytes（词元字节）的映射。

merges: list[tuple[bytes, bytes]] 训练产生的 BPE 合并列表。列表中的每个项都是一个字节元组（<词元1>，<词元2>），表示 <词元1> 已与 <词元2> 合并。合并应按创建顺序排序。
```

To test your BPE training function against our provided tests, you will first need to implement the test adapter at [adapters.run_train_bpe]. Then, run uv run pytest tests/test_train_bpe.py. Your implementation should be able to pass all tests. Optionally (this could be a large time-investment), you can implement the key parts of your training method using some systems language, for instance C++ (consider cppy for this) or Rust (using PyO3). If you do this, be aware of which operations require copying vs reading directly from Python memory, and make sure to leave build instructions, or make sure it builds using only pyproject.toml. Also note that the GPT-2 regex is not well-supported in most regex engines and will be too slow in most that do. We have verified that Oniguruma is reasonably fast and supports negative lookahead, but the regex package in Python is, if anything, even faster.

# Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)

(a) 在 TinyStories 数据集上训练一个字节级 BPE 分词器，最大词汇表大小为 10,000。确保将 TinyStories $<\text{endoftext}|$ 特殊词元添加到词汇表中。将生成的词汇表和合并序列化到磁盘以供进一步检查。训练花费了多少小时和多少内存？词汇表中 OOV 词元最长的是什么？这有意义吗？

资源要求：$\leq 30$ 分钟（无 GPU），$\leq 30$ GB RAM

提示：通过预分词过程中的多进程处理，您应该能够在 2 分钟内完成 BPE 训练，并利用以下两个事实：

(a) $<\text{endoftext}|$ 词元在数据文件中分隔文档。
(b) $<\text{endoftext}|>$ 词元在应用 BPE 合并之前被作为特殊情况处理。

交付成果：一到两句话的回答。

(b) 分析您的代码。分词器训练过程的哪个部分花费的时间最多？

交付成果：一到两句话的回答。

接下来，我们将尝试在 OpenWebText 数据集上训练一个字节级 BPE 分词器。与之前一样，我们建议您查看数据集以更好地了解其内容。

# 问题 (train_bpe_expts_owt): 在 OpenWebText 上训练 BPE（2 分）

(a) 在 OpenWebText 数据集上训练一个字节级 BPE 分词器，使用最大词汇表大小为 32,000。将生成的词汇表和合并保存到磁盘以供进一步检查。词汇表中存在的最长词元是什么？这是否合理？

资源需求：$\leq 12$ 小时（无 GPU），$\leq 100\mathrm{GB}$ RAM

交付物：一到两句话的回答。

(b) 比较和对比在 TinyStories 和 OpenWebText 上训练得到的 ist。

交付物：一到两句话的回答。

# 2.6 BPE 分词器：编码和解码

在本次作业的前一部分，我们实现了一个函数，用于在输入文本上训练 BPE 分词器，以获得分词器词汇表和 BPE 合并列表。现在，我们将实现一个 BPE 分词器，该分词器加载提供的词汇表和合并列表，并使用它们将文本编码/解码为词元 ID。

# 2.6.1 编码文本

BPE 编码文本的过程与我们训练 BPE 词汇表的方式类似。有几个主要步骤。

步骤 1：预分词。我们首先对序列进行预分词，并将每个预分词表示为 UTF-8 字节序列，就像我们在 BPE 训练中所做的那样。我们将把每个预分词内的这些字节合并到词汇表元素中，独立处理每个预分词（预分词边界之间没有合并）。

步骤 2：应用合并。然后，我们采用 BPE 训练过程中创建的词汇表元素合并序列，并按创建顺序将其应用于我们的预分词。

# 示例 (bpe_encoding)：BPE 编码示例

例如，假设我们的输入字符串是“the cat ate”，我们的词汇表是 $\{0: b' ', 1: b'a', 2: b' c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b'a', 9: b'the', 10: b' at'\}$，我们学习到的合并是 [(b't', b'h'), (b' ', b' c'), (b' ', a'), (b'th', b'e'), (b'a', b't')]。首先，我们的预分词器会将此字符串拆分为 ['the', 'cat', 'ate']。然后，我们将查看每个预分词并应用 BPE 合并。

第一个预词元 'the' 最初表示为 $[b't', b'h', b'e']$。查看我们的合并列表，我们确定第一个适用的合并是 $(b't', b'h')$，并用它将预词元转换为 $[b'th', b'e']$。然后，我们回到合并列表，确定下一个适用的合并是 $(b'th', b'e')$，它将预词元转换为 $[b'the']$。最后，回顾合并列表，我们发现没有更多的合并适用于该字符串（因为整个预词元已合并为单个词元），因此我们完成了 BPE 合并的应用。相应的整数序列是 [9]。

重复此过程以处理剩余的预分词，我们看到预分词 'cat' 在应用 BPE 合并后表示为 $[b'c', b'a', b't']$，它变成整数序列 [7, 1, 5]。最终的预分词 'ate' 在应用 BPE 合并后表示为 $[b'at', b'e']$，它变成整数序列 [10, 3]。因此，编码输入字符串的最终结果是 [9, 7, 1, 5, 10, 3]。

特殊标记。您的分词器应能够正确处理用户定义的特殊标记（在构建分词器时提供）。

内存考量。假设我们要对一个无法放入内存的大文本文件进行分词。为了有效地对这个大文件（或任何其他数据流）进行分词，我们需要将其分解成可管理的数据块，并依次处理每个数据块，这样内存复杂度就是常数，而不是文本大小的线性函数。在这样做的时候，我们需要确保一个词元不会跨越数据块边界，否则我们将得到与在内存中对整个序列进行分词的朴素方法不同的分词结果。

# 2.6.2 解码文本

要将整数词元 ID 的序列解码回原始文本，我们可以简单地在词汇表中查找每个 ID 对应的条目（字节序列），将它们连接在一起，然后将字节解码为 Unicode 字符串。请注意，输入 ID 不保证能映射到有效的 Unicode 字符串（因为用户可以输入任何整数 ID 序列）。如果输入词元 ID 未能生成有效的 Unicode 字符串，则应将格式错误的字节替换为官方 Unicode 替换字符 U+FFFD。$^3$ `bytesdecode` 的 `errors` 参数控制如何处理 Unicode 解码错误，使用 `errors='replace'` 将自动用替换标记替换格式错误的字节。

# 问题（分词器）：实现分词器（15 分）

交付物：实现一个 Tokenizer 类，该类接收一个词汇表和一组合并，将文本编码为整数 ID，并将整数 ID 解码为文本。您的分词器还应支持用户提供的特殊标记（如果它们尚不存在，则将它们附加到词汇表中）。我们建议使用以下接口：

```python
def __init__(self, vocab, merges, special_tokens=None)
```
从给定的词汇表、合并列表和（可选的）特殊标记列表构建一个分词器。此函数应接受

```python
以下参数：
vocab: dict[int, bytes]
merges: list[tuple[bytes, bytes]]
special_tokens: list[str] | None = None
```

```python
def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None) 类方法，用于从序列化的词汇表和合并列表（格式与BPE训练代码输出的格式相同）以及（可选的）特殊标记列表构建并返回一个分词器。此方法应接受以下附加参数：
```

```txt
vocab_filepath: str  
merges_filepath: str  
special_tokens: list[str] | None = None
```

```txt
def encode(self, text: str) -> list[int] 将输入文本编码为词元ID序列。
```

问题（tokenizer_experiments）：分词器实验（4分）
```txt
def encode iterable(self, iterable: Iterator[str]) -> Iterator[int] 给定一个字符串的可迭代对象（例如，Python文件句柄），返回一个生成器，该生成器惰性地产生词元ID。这对于内存效率高的分词处理无法直接加载到内存的大文件是必需的。
```

```python
defdecode(self，ids：list[int]）->strDecodeasequenceoftokenIDsinto text.
```

要测试您的分词器是否符合我们提供的测试，您首先需要实现 [adapters.get_tokenizer] 中的测试适配器。然后，运行 uv run pytest tests/test_tokenizer.py。您的实现应该能够通过所有测试。

# 2.7 实验

(a) 从 TinyStories 和 OpenWebText 中分别采样 10 个文档。使用您之前训练的 TinyStories 和 OpenWebText 分词器（词汇表大小分别为 10K 和 32K），将这些采样文档编码为整数 ID。每个分词器的压缩率（bytes/token）是多少？

交付物：一到两句话的回答。

(b) 如果您使用 TinyStories 分词器对 OpenWebText 样本进行分词，会发生什么？比较压缩率和/或定性描述发生了什么。

交付物：一到两句话的回答。

(c) 估算您的分词器的吞吐量（例如，每秒字节数）。对 Pile 数据集（825GB 文本）进行分词需要多长时间？

交付成果：一到两句话的回答。

(d) 使用您的 TinyStories 和 OpenWebText 分词器，将各自的训练和开发数据集编码为整数词元 ID 序列。我们稍后将使用它来训练我们的语言模型。我们建议将词元 ID 序列化为 uint16 数据类型的 NumPy 数组。为什么 uint16 是一个合适的选择？

![](data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCALSAlMDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKwNf8aeH/AAxNFBqt+I7iUbo4I43lkYeu1ATj3rfrz34fxre+LvGuq3Ch7wambRXbqkSKNqj0HegDr9D8Q6T4ksftukXsd1AG2sVyCjejKcFT7EVp15rITpHxj1dbHES3uhG6mVRwZUchXI9cU0eK9aPwFPiX7b/xN/sZl+0eUn3t+M7cbentQB6O1zAlxHbvPGs8gJSMuAzAdSB1OKlryDVbDWL34qeE5h4iuInurCaWMrbQnyQEUsoyvIb1OSO2K1LaXxV4n8X+KdOt/FEmmWWm3EaQCG0id8smcFmH3c8+pz1oA9LoryRvHPiAeDYGluo49Ws/EEelXk0cS7Zl34JCkEDII6fhiup8U6xqdp418L6TZXptrfUftKz4jRiSseVPzA4weffvQB1sVzBO8qQzxyNE2yQI4JRvQ46Gpa8t+GGnalB4q8XSXGuT3McWptHLG0Eaid9q/vCQuQfYYHtXqVABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUU2SSOGJ5ZXVI0UszscBQOpJ7CgB1FYsXjDwxPKsUPiPSJJGOFRL6Ikn2AatqgArhr3w34i0TxNf634Tk0+WPUtrXlhfs6L5ijAkRlBwSOoIruar319baZYz3t7MsNtAheSRuiqOpNAHLeHvCeoJqWqa74jubebVtRhFtstARFbQjoiFuTyckmuUl8D+Ov+EFn8EwTaENOCNHFevJL5jx7twUqFwp7E8/jXrFvPFdW0VxA4eGVA6OOjKRkGpKAOG1vwxro1zwzrOj/AGGa40uB7aeG5kZFZXUAspAPIweorntHbxRF8QPHMnh6LS51N3CskV9I8eG8vhgVByOuQcfWvTtT1Wx0az+16jcpb2+9U8x843McAfiadbadZWlzc3NtaxRT3TB55EUBpSBgFj34oA8/f4bagfAb6eNQt316TURqr3DKfJa43btvrtxx/SrCeHvGOseM9A13W/7GtrfSzLm3tZZHZt6bd2SuDzjjjA7muxuNe0m1tr64m1G2EVh/x9lZA3kHrhwOQfY1MmqWDz28C3sHn3MXnQRGQB5E/vKvUjmgDmPDvh/W9C8Za7Niyl0bVLg3Yk8xhNG+0Dbtxgjjrmuypk00VvC800iRxRqWd3bCqB1JJ6Cm21zBeW0dzazRzwSqGjljYMrg9CCOCKAJaKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACsvxN/yKusf9eU3/oBrUqhrltLeeH9StYE3zTWssca5AyxUgDJ460AeOeHtb+GQ+GVjZ6sNJmv/ALCEliW2DTtJjoCFzuz3rT0rXfGGheHPA2gWttatqWpQzq51PfmJVyyE4OeEIyMZ4xxXd+BdGm0XwVo1lf2iQ31vbKkq/KxVh1+YZB/A1V8Q6LqF94+8KanbW++zsDc/aZN6jZvQBeCcnJ9AaAKN34g8USazZ+FdNOmHWUsxdajfSxuYIgTgBEBBJJ9TWLr/AIk1K88IeNfDuvQ2yatp9gZPNtdwiuInHyuA2SD2IzW7rem61ovjn/hKtH006rDc2YtLy0jlWOVdrZV03EA+hGaxbrwt4h12w8X61eactrqWrWAsrLTvPRmSNRxvfO3cSc9eKAGQ+IvGPhbwnout6jFpMuhiK3jmtoVfz4YmCqH3k4Y8jIwP610XibWvENtqghsbnRNI0wRBxqGqyhvPc/wIgdSAO5P4Vzd/p/jHxD4V0/wbdeHBYxgQR3mom7jeLyo9pOwA7ix2jjHH61a1fQ9W0/x/eav/AMIrH4ks7m1ihtN00QNmVGCuJOgJ5yKAMTxR4ruPFfwcuL2WG2+222qxW0gtnzDK6SrhkPPynI9a6mXxF4t8Oa5pC+I10mfTdVuBag2SOr2srD5QSxO9T0zgVy6+CfFR+Ger6V/ZMEWpz60LyKCOdPK2b1bIOeFGDwcHjpXQX1t4n8Za3odtqPh46RYaZeLe3M8l1HKJXQHasYU5wSepxQBl+I9VOoeFfidbmxsrf7HJ5XmW8Ox5vlB3SHPzN2zWwmqXMXibwvpVpZ6abifQ3kiu7iAvJEwVcAMCCEJ6gdfWqV/4R12bSfiNBHY5k1icPYr5qfvhsA/vfLyP4sVrReHdVXxz4W1E2uLSx0h7a4k8xfkkIXC4zk9DyARQBzXgu78QW+j+ObrUJdMu7a3urwvC0Mh3zKoOOWx5RH8PX3rV/wCE4uoPDnhOx0u20u11TWbUSKJAY7W1jVQWO0HOOcBc0mmaJr9jF420V9HdotSkurq0vVmTZIZFAVME5Bz68Vn3/gHUTofg6+k0O11W60e0+zXmk3LoRIpUZ2scruUjI5waANvSvG+pW13rml6wtjqF5p1ib+GbTMhLiPkFdpJ2sDx171W8OeKfGOrw6Zq8X9happ146i5tLBis1mjdyzPhivcYB9Kt+HtIvIk1S50nwZpnhedrby7SSQRtLI/X5xHwEyBxnPFc83hzW9W1jSJ4PBcfh/WLa6jkvdXt7iJY5EH3wFQ5ff6Ecd6APYKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACisPxf4kg8I+Fr7W7hN62yDbHnBdyQqr+ZFa1ndwX9lb3lrIJLe4jWWJ16MrDII/A0ATUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAeBftIeI9sWleGoX5Ym8uAD2GVQf+hn8BXXfAjxH/AG38PY7GV91zpchtmyefLPzIfpglf+A1xnxp+GlyY9b8dXGuCXYYglkLXG1CyRhd+/tnPTk59a634SfDS58GP/a664Lq31KzjMlr9l2YY4ZTu3nOMsOnegD1WiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKK8y+Meq6gtnofhrTblrWTXrz7NLcKeViGAw/HePwBHegDs7jxf4ZtJ3gufEWkQzIcNHLexqyn0ILZFR/wDCceEf+hp0T/wYRf8AxVcXZ/CXwVaWscJ0ZJ2UYMs0jszn1POPywKn/wCFXeCv+hftv++n/wAajnRPOjrf+E48I/8AQ06J/wCDCL/4qj/hOPCP/Q06J/4MIv8A4quS/wCFXeCv+hftv++n/wAaP+FXeCv+hftv++n/AMaOdBzo63/hOPCP/Q06J/4MIv8A4qj/AITjwj/0NOif+DCL/wCKrkv+FXeCv+hftv8Avp/8aP8AhV3gr/oX7b/vp/8AGjnQc6Ot/wCE48I/9DTon/gwi/8AiqP+E48I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8Lz03/oVPE//AIBr/wDF16JRQB53/wALz03/AKFTxP8A+Aa//F1f0f41eE9Tv0sbs3uj3LkBV1KHywT/ALwJA/HFdrWT4g8N6T4o0ySw1a0SeJgdrEfPGf7yt1BoA6UEEAg5B6EUteX/AAj1K/sbjXPBGqTtPPoUqi2mbq9u33fy4PsGA7V6hQBzfjHxnY+CrC3vdQtL64gmk8sm0iDmPjO5skYFWPDPi/Q/F9ibvRL9LlUwJEwVeM+jKeR9eh7VuV458RdKg+H3iHSvH+hxLaqbpbbVLeEbUnjfq2Bxng/jtPUcgHsdFAIIyOQaKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACvJfD/APyX/wAbf9ett/6Ljr1qvJfD/wDyX7xv/wBett/6LjoA9KooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKb5ieZ5e9d+M7c84+lADqKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigDzjwp/wAnBeMf+vC3/wDQIq9Xryjwr/ycF4x/68Lf/wBAir1egArzD4+/8kuuP+vqH+den15h8ff+SXXH/X1D/OgD0uD/AI94v9wfyqSo4P8Aj3i/3B/KpKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACvJfD/wDyX7xv/wBett/6Ljr1qvJfD/8AyX7xv/1623/ouOgD0qiiigAooooAKKKKACiiigAooooAKKKKACiiigDnvEPgzw94il+2atpqXNxFEURzI64HJxwR3NcP8LPB3h6bwhpuvTaasmpRSySLOZHzuSRtvGccYHavVpgTBIAMkqcD8K434Z2N3YfDi1tby1nt7hTPmKaMo4y7EcHnmgCHw5461nX9POs/8I0YtHEcpLx3Pmzs6Z4SMKNwJGB0qC68deJdLs7bV9W8KxW2jzSIjYvd1xCHIAZk2gdxxmovDtpr+nfBQ29haz2+tJBN5MUkZSQMXYj5W746fhXB6vpNpqnhuI6X4d8UahriNE91dX6TMYsMN+AxwzE8YUHjNAHp1/4z1dvFd/4c0TQEvLq1ijmM812I4grDPzfKSDnAAGc89MVCnxIWPwhfaveaU8N/ZXf2Gay84Eef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igAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigDhPjKM/CXXv9yL/wBHJXR+EznwboZ9dPt//Ra1z3xj5+E2v/8AXOP/ANGpW/4POfBOgn/qHW//AKLWgDaooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA4b4xf8kn1/8A65R/+jUrf8Hf8iRoH/YNt/8A0WtYPxh/5JPr/wD1yT/0Ylbvg7/kR9A/7Btv/wCi1oA26KKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAoormPE/iS70TXfDVjbxQPHql6beYyAkqoXOVwRz9c0AdPRWPf+K/D2lmUX2uadbtE4jkWS5QFGIztIzkHHOKtSa1pUVnb3kmpWaWtwQIZmnUJKT0CtnBzjtQBblijnjaOaNJI26q65B/A05EWNFRFCqowFAwAKzNK8SaJrkkselavZXrxf6xbedXK+5APT3q1BqdhdW01zb3ttLBCzLLLHKrLGV+8GIOAR3z0oAtUVyGvfEjw/odzpETahYzrqMyr5iXkYEUbA4lPP3OOvT3ra1HxNoWkW0FxqOsWNrDcANC8s6qJAecrzyPcUAatFQ2t3bX1rHdWlxFcW8g3JLE4ZWHqCODU1ABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABXn/xA/wCRv8B/9hVv/QDXoFZ+oaJp2q3lhd3tv5s+ny+dbNvZfLfGM4BAPHrmgDhPCWkadffEbx5cXdjb3Eq3MMStLGHwpj5Az61xEMFvN8OfDllcoj2a+LDD5b8r5fmP8uPSvc7HRdP02/v760t/LudQdZLl97HzGUYBwTgcemK4nxj4Dt5tG0XSdH0rzLFdaS6u4fMLDYxYyMSzZxz0H4UAM8S2NlpnxR8Fy6Xbw295O08U6wIF3wBM/MB1APSqvgp1j+HXjDewXZfajuyenXrXYaH4H0Dw9qMmoWFpJ9sdPL86e4kmZU/uqXY4H0qpffDTwpqOp3F9c6c5kuW33EaXEiRSt/eZFYKT+FAHnkVpbS+EvhS8lvC7PeQozMgJZdrcH1HtXUa7o1xL8QJNS0FtF1G+t7BLe40nUMqYoycq0ZAO3PTkYrp73wR4fv8Aw7aaFNZMLCzKm2VJnV4SvQq+d2efWo9V8BeHtZa1kurWYXFtCII7iG5kil8sfws6sCw+uaAKXw2uNMm0K9j03S5NLaC+ljurJpfMWKfI3BD029MYwPauyrP0bRNN8Paamn6VapbWyEtsUkkk9SSckk+pNaFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAH//Z)  
图 1：我们的 Transformer 语言模型概述。

![](data:image/jpeg;base64,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)  
图 2：一个 Pre-norm Transformer 块。

# 3 Transformer 语言模型架构

语言模型接收一个整数词元 ID 的批处理序列（即，形状为 (batch_size, sequence_length) 的 torch.Tensor）作为输入，并返回一个（批处理的）词汇表上的归一化概率分布（即，形状为 (batch_size, sequence_length, vocab_size) 的 PyTorch Tensor），其中预测的分布是针对每个输入词元的下一个词。在训练语言模型时，我们使用这些下一个词的预测来计算实际下一个词与预测下一个词之间的交叉熵损失。在推理过程中从语言模型生成文本时，我们取最后一个时间步（即序列中的最后一个词元）的预测下一个词分布，以生成序列中的下一个词元（例如，通过选取概率最高的词元、从分布中采样等），将生成的词元添加到输入序列中，然后重复此过程。

在此次作业的这一部分，您将从头开始构建这个 Transformer 语言模型。我们将从模型的整体描述开始，然后逐步详细介绍各个组件。

# 3.1 Transformer LM

给定一个词元 ID 序列，Transformer 语言模型使用输入嵌入将词元 ID 转换为密集向量，将嵌入的词元通过 num_layers 个 Transformer 块，然后应用一个学习到的线性投影（“输出嵌入”或“LM 头”）来生成预测的下一个词元 logit。有关示意图，请参见图 1。

# 3.1.1 词元嵌入

在第一步中，Transformer 将（批处理后的）词元 ID 序列嵌入到包含词元身份信息的向量序列中（图 1 中的红色块）。

<output>
更具体地说，给定一个词元 ID 序列，Transformer 语言模型使用词元嵌入层来生成一个向量序列。每个嵌入层接收一个形状为 (batch_size, sequence_length) 的整数张量，并生成一个形状为 (batch_size, sequence_length, d_model) 的向量序列。

# 3.1.2 Pre-norm Transformer 块

嵌入后，激活值通过几个结构相同的神经网络层进行处理。标准的仅解码器 Transformer 语言模型由 num_layers 个相同的层（通常称为 Transformer“块”）组成。每个 Transformer 块接收形状为 (batch_size, sequence_length, d_model) 的输入，并返回形状为 (batch_size, sequence_length, d_model) 的输出。每个块通过自注意力机制聚合序列中的信息，并通过前馈层进行非线性变换。

# 3.2 输出归一化和嵌入
</output>

经过 num_layers 个 Transformer 块后，我们将获取最终的激活值，并将其转换为词汇表上的分布。

我们将实现“预归一化” Transformer 块（在 §3.5 中详述），该块还需要在最后一个 Transformer 块之后使用层归一化（下文详述），以确保其输出被正确缩放。

在此归一化之后，我们将使用标准的学习线性变换将 Transformer 块的输出转换为预测的下一个词元 logit（参见，例如，Radford et al. [2018] 方程 2）。

# 3.3 备注：批处理、Einsum 和高效计算

在整个 Transformer 中，我们将对许多类批处理输入执行相同的计算。以下是一些示例：

- 批次中的元素：我们对每个批次元素应用相同的 Transformer 前向操作。
- 序列长度：“逐位置”操作，如 RMSNorm 和前馈网络，对序列的每个位置执行相同的操作。  
- 注意力头：注意力操作在“多头”注意力操作中跨注意力头进行批处理。

以一种充分利用GPU且易于阅读和理解的方式执行此类操作非常有用。许多PyTorch操作可以接受张量开头的额外“类似批处理”的维度，并有效地跨这些维度重复/广播操作。

例如，假设我们正在执行一个逐位置的、批处理的操作。我们有一个形状为 (batch_size, sequence_length, d_model) 的“数据张量” $D$，我们希望与形状为 (d_model, d_model) 的矩阵 $A$ 进行批处理向量-矩阵乘法。在这种情况下，$D \otimes A$ 将执行批处理矩阵乘法，这是PyTorch中的一个高效的原始操作，其中 (batch_size, sequence_length) 维度被批处理。

因此，最好假设您的函数可能会获得额外的类似批处理的维度，并将这些维度保留在PyTorch形状的开头。为了以这种方式组织张量以便进行批处理，可能需要使用多个view、reshape和transpose步骤来对其进行塑形。这可能有点麻烦，而且很难看清代码的作用以及张量的形状。

一个更符合人体工程学的选择是在torch.einsum中使用einsum表示法，或者更确切地说，使用与框架无关的库，如einops或einx。两个关键操作是einsum，它可以对输入张量的任意维度进行张量收缩，以及rearrange，它可以重新排序、连接和分割任意

维度。事实证明，机器学习中的几乎所有操作都是维度调整和张量收缩的某种组合，偶尔会加上（通常是逐点）非线性函数。这意味着使用 einsum 表示法可以使您的许多代码更具可读性和灵活性。

我们强烈建议在课程中使用 einsum 表示法。以前没有接触过 einsum 表示法的学生应该使用 einops（文档在此处），而已经熟悉 einops 的学生应该学习更通用的 einx（在此处）。这两个包都已安装在我们提供的环境中。

这里我们给出了一些 einsum 表示法用法的示例。这些是对 einops 文档的补充，您应该先阅读它们。

示例（einstein_example1）：使用 einops.einsum 进行批量矩阵乘法。  
```python
import torch   
from einops import rearrange, einsum   
## 基本实现   
Y = D @ A.T   
# 很难说清输入和输出的形状以及它们的含义。   
# D 和 A 可以有什么形状，其中是否有任何形状会产生意外行为？   
## Einsum 是自文档化且健壮的   
# D A -> Y   
Y = einsum(D,A,"batch sequence d_in, d_out d_in -> batch sequence d_out")   
## 或者，一个批处理版本，其中 D 可以有任何前导维度，但 A 是受限的。   
Y = einsum(D,A,"... d_in, d_out d_in -> ... d_out")
```

示例 (einstein_example2): 使用 einops.rearrange 进行广播操作  
我们有一批图像，对于每张图像，我们想根据某个缩放因子生成 10 个变暗的版本：
images $=$ torch.rand(64, 128, 128, 3) # (batch, height, width, channel)
dim_by $=$ torch.linspace(start=0.0, end=1.0, steps=10)
重塑和相乘
dim_value $=$ rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")
images_rearr $=$ rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images $=$ images_rearr $*$ dim_value
# 或者一次性完成：
dimmed_images $=$ einsum(images, dim_by, "batch height width channel, dim_value -> batch dim_value height width channel")

示例 (einstein_example3): 使用 einops.rearrange 进行像素混合  
假设我们有一个由形状为 (batch, height, width, channel) 的张量表示的图像批次，我们想要对图像的所有像素执行线性变换，但此变换应独立于每个通道进行。我们的线性变换由一个形状为 (height $\times$ width, height $\times$ width) 的矩阵 $B$ 表示。 channels_last = torch.randn(64, 32, 32, 3) # (batch, height, width, channel) B = torch.randn(32*32, 32*32) #将图像张量重新排列以跨所有像素进行混合 channels_last_flat = channels_last.view(-1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)) channels_first_flat $\equiv$ channels_last_flat.transpose(1, 2) channels_first_flat_transformed $\equiv$ channels_first_flat @ B.T channels_last_flat_transformed $\equiv$ channels_first_flat_transformed.transpose(1, 2) channels_last_transformed $\equiv$ channels_last_flat_transformed.view(*channels_last.shape)改用 einops：
height $=$ width $= 32$
# Rearrange 取代了笨拙的 torch view + transpose
channels_first $\equiv$ rearrange( channels_last, "batch height width channel -> batch channel (height width)"
)
channels_firsttransformed $\equiv$ einsum( channels_first, B, "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"
)
channels_lasttransformed $\equiv$ rearrange( channels_first_transformed, "batch channel (height width) -> batch height width channel", height=height, width=width
)
或者，如果你想更进一步：使用 einx.dot（einx 相当于 einops.einsum）一次性完成
height $=$ width $= 32$
channels_lasttransformed $\equiv$ einx.dot( "batch row_in col_in channel, (row_out col_out) (row_in col_in)" "-> batch row_out col_out channel", channels_last, B, col_in=width, col_out=width
```

这里的第一个实现可以通过在前后放置注释来指示来改进
```

which the input and output shapes are, but this is clunky and susceptible to bugs. With einsum notation, documentation is implementation!

Einsum notation can handle arbitrary input batching dimensions, but also has the key benefit of being self-documenting. It's much clearer what the relevant shapes of your input and output tensors are in code that uses einsum notation. For the remaining tensors, you can consider using Tensor type hints, for instance using the jaxtyping library (not specific to Jax).

We will talk more about the performance implications of using eaxsum notation in assignment 2, but for now know that they're almost always better than the alternative!

# 3.3.1 Mathematical Notation and Memory Ordering

Many machine learning papers use row vectors in their notation, which result in representations that mesh well with the row-major memory ordering used by default in NumPy and PyTorch. With row vectors, a linear transformation looks like

$$
y = x W ^ {\top}, \tag {1}
$$

对于行主序 $W \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}$ 和行向量 $x \in \mathbb{R}^{1 \times d_{\mathrm{in}}}$。

在线性代数中，使用列向量通常更常见，此时线性变换看起来像

$$
y = W x, \tag {2}
$$

给定行主序 $W \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}$ 和列向量 $x \in \mathbb{R}^{d_{\mathrm{in}}}$。在本作业中，我们将使用列向量进行数学表示，因为这样通常更容易理解数学。您应该记住，如果您想使用纯矩阵乘法表示法，您将不得不应用矩阵的行向量约定，因为 PyTorch 使用行主序内存。如果您使用 `einsum` 进行矩阵运算，这将不是问题。

# 3.4 基本构建块：线性（Linear）和嵌入（Embedding）模块

# 3.4.1 参数初始化

有效训练神经网络通常需要仔细初始化模型参数——糟糕的初始化可能导致梯度消失或爆炸等不良行为。预归一化 Transformer 对初始化异常鲁棒，但它们仍然可能对训练速度和收敛产生重大影响。由于本次作业已经很长了，我们将把细节留到作业 3，而是提供一些近似初始化方法，这些方法在大多数情况下都应该效果良好。目前，请使用：

- 线性权重： $\mathcal{N}\left(\mu = 0, \sigma^2 = \frac{2}{d_{\mathrm{in}} + d_{\mathrm{out}}}\right)$ ，截断在 $[-3\sigma, 3\sigma]$ 。
- 嵌入： $\mathcal{N}\left(\mu = 0, \sigma^2 = 1\right)$ ，截断在 $[-3, 3]$ 。
RMSNorm：1

您应该使用 torch(nn.init.trunc_normal_) 来初始化截断正态权重。

# 3.4.2 线性模块

线性层是 Transformer 和神经网络的基本组成部分。首先，您将实现自己的 Linear 类，该类继承自 torch.nnModule 并执行线性变换：

$$
y = W x. \tag {3}
$$

请注意，我们不包含偏置项，这与大多数现代 LLM 保持一致。

# 问题 (linear)：实现线性模块 (1 分)

交付成果：实现一个继承自 torch(nn.Module 并执行线性变换的 Linear 类。您的实现应遵循 PyTorch 内置 nn.Linear 模块的接口，但没有偏置参数或参数。我们推荐以下接口：

def __init__(self, in_features, out_features, device=None, dtype=None) 构建一个线性变换模块。此函数应接受以下参数：

in_features: int 输入的最终维度

out_features: int 输出的最终维度

device: torch_device | None = None 用于存储参数的设备


 dtype: torch.dtype | None = None 参数的数据类型

def forward(self, x: torch.Tensor) -> torch.Tensor 对输入应用线性变换。

请确保：

- 继承 nn-module
- 调用父类构造函数
- 为了内存排序原因，将参数构建并存储为 $W$（而不是 $W^{\top}$），并将其放入 nn_PARAMETER
- 当然，不要使用 nn.Linear 或 nn.Functional-linear

对于初始化，请使用上述设置，并结合 torch(nn.init.trunc_normal_ 来初始化权重。

要测试你的 Linear 模块，请在 [adapters.runlinear] 实现测试适配器。该适配器应将给定的权重加载到你的 Linear 模块中。你可以为此目的使用 Module.load_state_dict。然后，运行 uv run pytest -k testlinear。

# 3.4.3 Embedding Module

如上所述，Transformer 的第一层是一个嵌入层，它将整数词元 ID 映射到一个维度为 d_model 的向量空间。我们将实现一个自定义的 Embedding 类，该类继承自 torch.nnModule（因此你不应使用 nn.Embedding）。forward 方法应通过索引形状为 (vocab_size, d_model) 的嵌入矩阵来选择每个词元 ID 的嵌入向量，其中使用形状为 (batch_size, sequence_length) 的词元 ID 的 torch.LongTensor。

# 问题（嵌入）：实现嵌入模块（1 分）

交付物：实现继承自 torch(nnModule 并执行嵌入查找的 Embedding 类。你的实现应遵循 PyTorch 内置 nn.Embedding 模块的接口。我们推荐以下接口：

def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) 构建一个嵌入模块。此函数应接受以下参数：

num_embedding：int 词汇表的大小


embedding_dim: int embedding向量的维度，即 $d_{\mathrm{model}}$

device: torch_device | None = None 参数存储的设备

dtype: torch.dtype | None = None 参数的数据类型

def forward(self, token_ids: torch.Tensor) -> torch.Tensor 查找给定词元ID的embedding向量。

确保：

- 继承 nn-module
- 调用父类构造函数
- 将embedding矩阵初始化为 nn_PARAMETER
- 存储embedding矩阵，其中 d_model 是最终维度
- 当然，不要使用 nn.Embedding 或 nnfunctional_embedding

再次，使用上面的设置进行初始化，并使用 torch.nn.init.trunc_normal_ 来初始化权重。

为了测试您的实现，请在 [adapters.run_embedding] 中实现测试适配器。然后，运行 uv run pytest -k test_embedding。

# 3.5 Pre-norm Transformer 块

每个 Transformer 块包含两个子层：一个多头自注意力机制和一个逐位置前馈网络（Vaswani et al., 2017, section 3.1）。

<output>
在原始的Transformer论文中，模型在两个子层周围都使用了残差连接，然后进行层归一化。这种架构通常被称为“后归一化”（post-norm）Transformer，因为层归一化应用于子层输出。然而，大量研究发现，将层归一化从每个子层的输出移到每个子层的输入（在最后一个Transformer块之后增加一个层归一化）可以提高Transformer的训练稳定性 [Nguyen and Salazar, 2019, Xiong et al., 2020]—有关这种“前归一化”（pre-norm）Transformer块的视觉表示，请参见图 2。然后，通过残差连接将每个Transformer块子层的输出添加到子层输入（Vaswani et al., 2017, section 5.4）。前归一化的直观理解是，存在一个没有归一化的干净的“残差流”从输入嵌入流向Transformer的最终输出，这被认为可以改善梯度
</output>flow。这种 Pre-norm Transformer 块现在是当今语言模型（例如 GPT-3、LLaMA、PaLM 等）的标准，因此我们将实现此变体。我们将逐步介绍 Pre-norm Transformer 块的每个组件，并按顺序实现它们。

# 3.5.1 均方根层归一化

Vaswani 等人 [2017] 的原始 Transformer 实现使用层归一化 [Ba et al., 2016] 来归一化激活。遵循 Touvron 等人 [2023]，我们将使用均方根层归一化（RMSNorm；Zhang and Senrich, 2019, 方程 4）进行层归一化。给定一个激活向量 $a \in \mathbb{R}^{d_{\mathrm{model}}}$，RMSNorm 将按如下方式重新缩放每个激活 $a_i$：

$$
\operatorname {R M S N o r m} \left(a _ {i}\right) = \frac {a _ {i}}{\operatorname {R M S} (a)} g _ {i}, \tag {4}
$$

其中 $\mathrm{RMS}(a) = \sqrt{\frac{1}{d_{\mathrm{model}}}\sum_{i = 1}^{d_{\mathrm{model}}}a_i^2 + \varepsilon}$ 。这里，$g_{i}$ 是一个可学习的“增益”参数（总共有 $d_{\mathrm{model}}$ 个这样的参数），而 $\varepsilon$ 是一个超参数，通常固定为 1e-5。

您应该将输入上转换为 torch.float32，以防止输入平方时发生溢出。总体而言，您的 forward 方法应如下所示：

```txt
in dtype  $=$  x.dtype   
 $\mathbf{x} = \mathbf{x}$  to(torch.float32)
```

您的代码在此处执行 RMSNorm

```txt
... result  $=$  ...
```

以原始 dtype 返回结果

return result.to(indtype)

# 问题 (rmsnorm)：均方根层归一化（1 分）

交付成果：将 RMSNorm 实现为 torch.nnModule。我们推荐以下接口：

```python
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) 构建 RMSNorm 模块。此函数应接受以下参数：
```

d_model: int 模型的隐藏层维度

eps: float = 1e-5 数值稳定性使用的 epsilon 值

device: torch_device | None = None 用于存储参数的设备

dtype: torch.dtype | None = None 参数的数据类型

def forward(self, x: torch.Tensor) -> torch.Tensor 处理形状为

(batch_size, sequence_length, d_model) 的输入张量并返回相同形状的张量。

注意：请记住，在执行归一化（以及稍后向下转换为原始 dtype）之前，将输入上转换为 torch.float32，如上所述。

要测试您的实现，请在 [adapters.run_rmsnorm] 中实现测试适配器。然后，运行 uv run pytest -k test_rmsnorm。

# 3.5.2 位置前馈网络

![](data:image/jpeg;base64,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voaVfuD6Uj/6tvoaVfuD6UALRRRQAUUUUAFFFFABRRRQAUUUUAFFFFACH7p+lJH/AKpP90Up+6fpSR/6pP8AdFADqKKKACiiigAooooAKKKKACiiigAooooAZN/qZP8AdP8AKqWhf8gOz/65Crs3+pk/3T/KqWhf8gOz/wCuQoA0KKKKACiiigAooooAKKKKACiiigAooooAKZF/qlp9Mi/1S0APoopk00VvC800ixxRqWd2OAoHUk0AOJCgkkADkk9q821SDVviDr7aeoktPDVq+XmH/L0Qf4T36HHYdTzgUsOs6v488SomkSvaeH7KT99OV/4+T3Ug9QQfunoDk84FUvG3ia81jU08AeDQPtLjy725j4S2jHBXI6YHU9ug5PF4PCSzGpyRdorVvpZbt+X5nNKVOvTe+/yf67mdrl/N401NPAHgzbbaLagLqF7GPk2g8qD3Gc/7x9gSfVdB0Kw8N6PBpemwiO3hH/AnbuzHuTVTwl4U0/wfocem2C5P3ppiPmmfux/oOwrdruxuLhKKw2GVqUdu7f8AM/07I2hC2rCiiivOLCiiigAooooAKZF/q/xP86fTIv8AV/if50APooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKhuLu2tED3NxFChOA0jhR+tAE1Y/i3/kTNd/7B9x/6LatZHSVFeNldGGQynIIrJ8W/wDIma7/ANg+4/8ARbUAaFj/AMg+2/65L/IVYqvY/wDIPtv+uS/yFWKACiiigAooooAa/wDq2+hpV+4PpSP/AKtvoaVfuD6UALRRRQAUUUUAFFFFABRRRQAUUUUAFFFFACH7p+lJH/qk/wB0Up+6fpSR/wCqT/dFADqKKKACiiigAooooAKKKKACiiigAooooAZN/qZP90/yqloX/IDs/wDrkKuzf6mT/dP8qpaF/wAgOz/65CgDQooooAKKKKACiiigAooooAKKKKACiiigApkX+qWn1ErpHb75GVUUEszHAA9TQA6WWOCJ5ZXVI0UszscBQOpJribbWY/iFc3enW8LpokDYmnZf+Pn0VePl559cYPFTeLNG1XxdLY2ljfwx6DJlrqSJsuxB6e49Pfr0FZ3jDxRaeAdHtfDvhy2EusXIEdpbRjcUyceYw7knpnqfYGnh8NWxldUaa0/r7kurM/aVIVHde6u6ve/+X5lPxj4m/sCK18DeCoA2s3CiMCL/l2U9WJ/vkZOT0HzHtXS+BPBFp4L0fyVIn1CfD3d0esjeg/2R2/PvVL4e+BB4XtZdR1J/tWv33z3Vwx3FcnJQH69T3P0FdxXp4vEUqVL6nhX7i+J/wAz/wDkV0XzCEer/wCGCiiivLNAooooAKKKKACiiigApkX+r/E/zp9Mi/1f4n+dAD6KKKACiiigAooooAKKKKACiiigAooooAKKKKACuU1P4b+F9b1641nV9PN/dTBVAnlYpGAoGFUEAdM9+Sa6uvN/H3j42mqp4Q0O/tLTWLhQbi9upVSKwiIzuJY8uQQQvuD6UAUfANpDofxW8T6BoLyHw9BbRyvDvLpb3JI+VSScZG7P0x2ruPGV1bweENajlnijd9PnCK7gFv3bdB3qj4GtPC2i6YNI0DVbO+n5nuZI7lJZp3ON0j4JPUj6cCrfjXT7K78J6xNc2dvNLFp85jeSJWZD5ZPBI4oA2LFl/s+25H+qXv7CrG5f7w/Oq1iif2fbfKv+qXt7CrGxP7i/lQAu5f7w/Ojcv94fnSbE/uL+VGxP7i/lQAu5f7w/Ojcv94fnSbE/uL+VGxP7i/lQAjsvltyOh70qsuwcjp6010Ty2+Veh7UqomwfKvT0oAduX+8Pzo3L/eH50mxP7i/lRsT+4v5UALuX+8Pzo3L/AHh+dJsT+4v5UbE/uL+VAC7l/vD86Ny/3h+dJsT+4v5UbE/uL+VAC7l/vD86Ny/3h+dJsT+4v5UbE/uL+VAC7l/vD86Ny/3h+dJsT+4v5UbE/uL+VAC7l/vD86Ny/wB4fnSbE/uL+VGxP7i/lQAFl2n5h09aSNl8pOR0Hegom0/Iv5UkaJ5SfKvQdqAH7l/vD86Ny/3h+dJsT+4v5UbE/uL+VAC7l/vD86Ny/wB4fnSbE/uL+VGxP7i/lQAu5f7w/Ojcv94fnSbE/uL+VGxP7i/lQAu5f7w/Ojcv94fnSbE/uL+VGxP7i/lQAu5f7w/Ojcv94fnSbE/uL+VGxP7i/lQAu5f7w/Ojcv8AeH50mxP7i/lRsT+4v5UANmZfJk+YfdPf2qlobL/YdnyP9UO9XJkTyZPkX7p7VS0NE/sOz+Vf9UO1AGjuX+8Pzo3L/eH50mxP7i/lRsT+4v5UALuX+8Pzo3L/AHh+dJsT+4v5UbE/uL+VAC7l/vD86Ny/3h+dJsT+4v5UbE/uL+VAC7l/vD86Ny/3h+dJsT+4v5UbE/uL+VAC7l/vD86Ny/3h+dJsT+4v5UbE/uL+VAC7l/vD86Ny/wB4fnSbE/uL+VIwiRSzBFVRkk4AAoAduX+8Pzry7UdRv/iJqjaHpMjW+h27f6Zd4x5vPQf0H4mtSHxLN4u8RzaTpNtu0SNCl1eqdhJ7bD+GMdwT061qa/ruieAPC7XckEUa9ILeMANPJjp/ix7VnTpTxk1Spa3dtOv/AADCfJXp3UtL66br1KfivxTpnw48L29raJ512U8mxtNxZnP95u+ATz69BWf8PPBVzZXEvirxM/2jxFfZf95z9mU9h6Njj2HA75peAvCN9rOrHxx4tQPfz/NZWjL8tun8J29jjoO3U8nj1HYn9xfyr3MRUp4Kk8Jh3eT+OS/9JXkuvd+RcVzav5C7l/vD86Ny/wB4fnSbE/uL+VGxP7i/lXjmgu5f7w/Ojcv94fnSbE/uL+VGxP7i/lQAu5f7w/Ojcv8AeH50mxP7i/lRsT+4v5UALuX+8Pzo3L/eH50mxP7i/lRsT+4v5UALuX+8Pzo3L/eH50mxP7i/lRsT+4v5UALuX+8PzpkTLs6jqe/vTtif3F/KmRImz7q9T296AJNy/wB4fnRuX+8PzpNif3F/KjYn9xfyoAXcv94fnRuX+8PzpNif3F/KjYn9xfyoAXcv94fnRuX+8PzpNif3F/KjYn9xfyoAdnPSigAAYAwKKACiiigAooooAKKKKACsPUPBvhnVr6S91DQNNurqTG+aa2R3bAAGSR6AD8K3KKAMjS/Cvh/Q7prrStFsLKdkMbSW9uqMVJBIyB0yB+VJ4t/5EzXf+wfcf+i2rYrH8W/8iZrv/YPuP/RbUAaFj/yD7b/rkv8AIVYqvY/8g+2/65L/ACFWKACiiigAooooAa/+rb6GlX7g+lI/+rb6GlX7g+lAC0UUUAFFFFABRRRQAUUUUAFFFFABRRRQAh+6fpSR/wCqT/dFKfun6Ukf+qT/AHRQA6iiigAooooAKKKKACiiigAooooAKKKKAGTf6mT/AHT/ACqloX/IDs/+uQq7N/qZP90/yqloX/IDs/8ArkKANCiiigAooooAKKKKACiiigAooooARmCqWYgADJJ7VxeqXKeP7WfSdGvWjsUbF1eJ0JH8AH8QPX06fjV8XnX/ABFrY8L6bBLaaeUD3d8y/K6HsD3HbHUn0ANdHb2+i+B/DTnclrYWqb5ZX6sfU+rHgfkB2rFKVafs4rTb1fZGUZ80pRlH3bNa338iGaXQvh54TaV8W9lbL9Xmc/8AoTH/ADgCuC8IeH7/AOIOup4z8URbdPjP/Es09uV2g8MR3Hf/AGjz0ABh0rT774veIl13V4pLfwrZORZ2bHH2gg8k/wBT/wABHc17DbosduiIoVFGFVRgAegr6CbjldJ0af8AFatJr7K/lXn3fTZBGKla2yJKKKK8U1CiiigAooooAKKKKACiiigAooooAKZF/q/xP86fTIv9X+J/nQA+iiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKx/Fv/ACJmu/8AYPuP/RbVsVj+Lf8AkTNd/wCwfcf+i2oA0LH/AJB9t/1yX+QqxVex/wCQfbf9cl/kKsUAFFFFABRRRQA1/wDVt9DSr9wfSkf/AFbfQ0q/cH0oAWiiigAooooAKKKKACiiigAooooAKKKKAEP3T9KSP/VJ/uilP3T9KSP/AFSf7ooAdRRRQAUUUUAFFFFABRRRQAUUUUAFFFFADJv9TJ/un+VUtC/5Adn/ANchV2b/AFMn+6f5VS0L/kB2f/XIUAaFFFFABRRRQAUUUUAFFFFAATgZNeb6x4j1Pxfrn9geFZ2itYWBvNRToBnop9P/AEL6ZJ15PEVr4sv77w5pcrbETFxdhN0bIch1U+vYE8HnHTnd0jRtM8MaUbezRYLZMySSO3J9WZj7fkBWEr1XaL06/wCRjNOrFOEtNb230LbywaZpplubgJBbRZkmlboqjkkmvIf9O+M3iT/ltbeDdPl91a6cf1P/AI6D6ml1O/1D4weIW0bSZJLbwpZSA3d2Bg3DDoB/Qf8AAj2Fet6Zplno+mwafYQLBawLsjjXsP6nuT3r6JJZXC7/AI7Wn9xPr/ifTsim/aPy/MmtraCytYrW2iSGCJQkcaDAVR0AFOi/1S0+mRf6pa8Vtt3ZoPooopAFFFFABRRRQAUUUUAFFFFABRRRQAUyL/V/if50+mRf6v8AE/zoAfRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFY/i3/kTNd/7B9x/6Latisfxb/yJmu/9g+4/9FtQBoWP/IPtv+uS/wAhViq9j/yD7b/rkv8AIVYoAKKKKACiiigBr/6tvoaVfuD6Uj/6tvoaVfuD6UALRRRQAUUUUAFFFFABRRRQAUUUUAFFFFACH7p+lJH/AKpP90Up+6fpSR/6pP8AdFADqKKKACiiigAooooAKKKKACiiigAooooAZN/qZP8AdP8AKqWhf8gOz/65Crs3+pk/3T/KqWhf8gOz/wCuQoA0KKKKACiiigAooooAK4vxZLqniIf2J4dnRUYgX1yDwiH+EN375A57etZ3ijxDqHiPV38JeGiQ2St9ecgRr0Zc/oT36Cuy0DRLbw7osGnWzMyRDLO55dj1Pt9KwcvatwW3V/oYxnCrzws7bXvbX9Rnh/w/YeGNJWzs1Cqo3Syt96Ru7Mf84rzPxBrWofFLXn8K+Gpmi0KBgdR1BfuyDPQeo44H8R56DNSeKPEeo/ETXJPB3hOXZpyHGpakv3dvdQe47f7R46ZJ9I8N+HNO8LaNDpmmxbIk5Zz96Ru7Me5P/wBbpX0NGnDK6aqzX71/DH+VfzPz7L5sEk1yx0iiXQ9EsPDukQaZpsAitoRgDux7sx7k+taNFFePOcpycpO7ZtsFMi/1S0+mRf6pakB9FFFABRRRQAUUUUAFFFFABRRRQAUUUUAFMi/1f4n+dPpkX+r/ABP86AH0UUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABWP4t/5EzXf+wfcf+i2rYrH8W/8iZrv/YPuP/RbUAaFj/yD7b/rkv8AIVYqvY/8g+2/65L/ACFWKACiiigAooooAa/+rb6GlX7g+lI/+rb6GlX7g+lAC0UUUAFFFFABRRRQAUUUUAFFFFABRRRQAh+6fpSR/wCqT/dFKfun6Ukf+qT/AHRQA6iiigAooooAKKKKACiiigAooooAKKKKAGTf6mT/AHT/ACqloX/IDs/+uQq7N/qZP90/yqloX/IDs/8ArkKANCiiigAooooAK4jUPGl1e+KoNA8Nwx3Lo/8AplywykSg84+nc/gOav61fSeILa50fQL1RcbjDc3Mbj/RsYyD3yenHvzkcXPDPhmw8KaSLW2AZyN09www0jep9AOw7fmaxk5Tlyx26v8ARGc1UvFwatvfR7Pa3+Zqw2drayzTQW8UUk7bpXRAC59Se9eVeKPFGpePdak8HeDpMWo41HU1+4E6FVI7duPvdBxklPEninU/iDrEnhLwbIVsV+XUNUH3dvQqp9Oo45btxkn0Lwt4W03wjo0em6bHhR80srfflfuzH/OK+gpUoZbBVqyvVesY/wAv96Xn2XzYN8+i2F8L+F9N8JaLHpumx4UfNJK335X7sx/zitqiivIqVJ1Zuc3dvdlpW0QUUUVAwpkX+qWn0yL/AFS0APooooAKKKKACiiigAooooAKKKKACiiigApkX+r/ABP86fTIv9X+J/nQA+iiigAooooAKKKKACiiigArgr3xt4g1O6mtfB/heW9SJzG2o6g/2e33A4O0H5pBnjIxXe1j+JtT1LR9Ekv9L0ttTmhZWe1jbDvHn5tvqwHIHfGKAOQEnxihPnPB4RnXvbo06t9ATx+prZ0Dxle3mqR6P4g8PXmjanIGMRJE1vNtGTslXjOATg1Afit4S/4Roa1/aOVLeWLPb/pPm/8APPy+u79PfHNbXhbUtW1fR/t2saX/AGZLLIzQ2rNl0i427/RjySOMUAbdFFFABRRRQAVj+Lf+RM13/sH3H/otq2Kx/Fv/ACJmu/8AYPuP/RbUAaFj/wAg+2/65L/IVYqvY/8AIPtv+uS/yFWKACiiigAooooAa/8Aq2+hpV+4PpSP/q2+hpV+4PpQAtFFFABRRRQAUUUUAFFFFABRRRQAUUUUAIfun6Ukf+qT/dFKfun6Ukf+qT/dFADqKKKACiiigAooooAKKKKACiiigAooooAZN/qZP90/yqloX/IDs/8ArkKuzf6mT/dP8qpaF/yA7P8A65CgDQooooAK4fxprOsXN5F4a8OxN9quR+/uweIF7jI+6cc59OmSeIfFXi29u9UHhjwsfM1Jzie5X7tuO/PYjue3Tr06S2Sx8JeHjNqF3FGkS77q7fjzH7sepJJ6D8BWOteXs6f4fkjHmhWjOCbTVtVt56kPhrw3p3g7RmjR03bfMurqTC7iBySeyjnjt+Zrz3WfEOr/ABS1WXw54Vd7bQoztv8AUyCPMX+6PY9h1bvgZqOa6134yag1rYmbTPB8MmJZyMPdEHp7/ToOpycCvVtF0XT/AA/pcOm6ZbrBbRDhR1J7knuT619BClSyqKcknW6LdQ833l2XTdhGKtyx0iiDw34a03wro8em6ZDsiXl3PLyt3Zj3P/6hWvRRXkVKk6knObu3uzZK2iCiiioAKKKKACmRf6pafTIv9UtAD6KKKACiiigAooooAKKKKACiiigAooooAKZF/q/xP86fTIv9X+J/nQA+iiigAooooAKKKKACiiigArH8TXur2GiSSaDpy3+pO6xwxSPtRSxxvY/3R1PStiuCvdd8c+GbmX7ZoCeIdNLsY7nTG2XCJngPEfvHtlaAMD/hUOqRTDxPFryv41EpuGneFfsrNjHl7MZAxxu698enoPhbVdU1bR/N1rSm0zUYpGhng3blLDHzIe6nOR/M1yY+LglPlW/gfxfJddPKbTgoB9zuOPyrZ0C98aaxqcV7qunWmh6Sgb/QjJ59zMSMAswwqAdcDnjBoA66iiigAooooAKx/Fv/ACJmu/8AYPuP/RbVsVj+Lf8AkTNd/wCwfcf+i2oA0LH/AJB9t/1yX+QqxVex/wCQfbf9cl/kKsUAFFFFABRRRQA1/wDVt9DSr9wfSkf/AFbfQ0q/cH0oAWiiigAooooAKKKKACiiigAooooAKKKKAEP3T9KSP/VJ/uilP3T9KSP/AFSf7ooAdRRRQAUUUUAFFFFABRRRQAUUUUAFFFFADJv9TJ/un+VUtC/5Adn/ANchV2b/AFMn+6f5VS0L/kB2f/XIUAaFc5L4wsG8VReHbWOS6uWz5zx/chwOQT6/568VF4v1HVTbDSvDqh9TnHztnHkx9C2egOf698V5+niTT/BSHQPCkH9veKro7Z54wWRX7gkdQPQH1JI6VeHw2Ixdb2VCO276JeuyMqsqkJxVtN7vqvI7C6n8KfCnSJ7l2YS3DFlQsHnnPZR7D8h35Ncrp/h3XvinqEWs+KhJYeH4232mmIxUyjsT3wf7x5PbAOa2PC3wzmfUh4i8aXP9q60+GWJzuig9BjoSPQfKOwPWvSq9X29DLo+zwfvT6z6Lyh/8lv2FGF1a1l2IbS0t7C0itbSCOC3iULHHGuFUegFTUUV5DbbuzYKKKKQBRRRQAUUUUAFMi/1S0+mRf6paAH0UUUAFFFFABRRRQAUUUUAFFFFABRRRQAUyL/V/if50+mRf6v8AE/zoAfRRRQAUUUUAFFFFABRRRQAVwGra18TINWuotL8J6bc2CSEQTSXqqzp2JG7g139cnf8AxN8GaXfz2N7r1vDdW7mOWNlfKsOo4FAHPt4o+KloPOufAdjPCnLpbagvmEe3J5/A12fhXxLaeLPD8GrWaSRLIWSSGUYeKRThkYeoNc9c/GPwFbQNKfEEUm0cJFFIzN7AbaPhVbXK+G9Q1O5tntf7Y1S51KK3kXDRxyEbQR9Fz+NAHdUUUUAFef8AiDxB4g1bxz/wh/hi7g09re1F1f6jLCJmiBI2oiHgk5B57Htjn0CvNPCw8n45+OI5uJJra0liz3QRgHH4kCgC94Z8Q67Y+Nbnwb4nuIL25+yi9stQhi8rz4920h0HAYH07A10vi3/AJEzXf8AsH3H/otq4zWh537QXhpIuXt9Knlmx2Rt6rn/AIFXReP7K+uPCOsTWmqXNqItOuC0EccRWY+WxwxZSR6cEdaAOhsf+Qfbf9cl/kKsV88x6d8XFt4yiawU2Dbt1BDxjjo1adlqnxW06wktZNF1G5LFj50kod1yMcHnpXZTy7EzdrRX/b8P/kjnjWk3ZxaPc6K+bnl+KqdbfxKf913P8q17Hxf4/wBP0eWyu/DviSaVg4F0UcsuRx1jJ4/3vyq4ZVjZP4F/4HD/AOSCNdt6xaPeqK+ZT4g8ax/8fF14si/7Yv8A1Ires/iZc2WgSWF7N4lNyyOoumtk3KWzg8tnjPr27VUcnzGX/Lr8Yv8AJhHEX3jY96f/AFbfQ0q/cH0r5mTxwzZWTxr4mjPcPbD+k1dZY/EnRl8OHTpvGOrC7Mbr9qeyJYEkkHgk8ZHftU/2RmS3oS/P8ghXvurfNHt1FfP0Wv2UvH/C3NQQ+j2dwP6111t4l0eXw5/Zw+JS/ayhX7XIuxsk5z83Pt1rD6jjl8dCS+THCs5bq3zR6nRXjkOnTXH+q+McLn0F0ufy82uwi028u/Dg0638bNLd7Av2uMqzZznPBz7dawdOvG/PTa9RxqSle8bfNHZ0V5v/AMK+8Wf9FB1D/vhv/i635/DWtSeHBpyeKbtLsIq/axGN2QQSeuecevesozk73iOM5u942+46mivN/wDhX3iz/ooOof8AfDf/ABdb934a1qfw8mnxeKbuK6VEU3QjG4lcZPBzzg9+9CnJ3vEIzm73jb7jqaK83/4V94s/6KDqH/fDf/F1v3/hrWrrQo7GDxTdwXKqgNysYyxHU8EHn60Kcne8QjObTvG33HU0V5v/AMK+8Wf9FB1D/vhv/i639U8Na1faPFZ23im7tp027p1jGWwOehB5+tCnJp3iEZzad4/kdQfun6Ukf+qT/dFecH4feLMH/i4Oof8AfDf/ABdb2q+G9a1HSILa08UXVnMpVmmSMZYBSCOCD3z17UKcmneIRnNptx/I6uivN/8AhX3iz/ooOof98N/8XW/rvhrWtTsYoLPxTd2UiOGaRIwCwwRj5SD3/ShTk07xBTm024/kdTRXm6/D/wAWBgT8QNQIB6bG/wDi63tf8M63qttFHZeKbuxdH3M6Rgbhjp8pFCnJptxBTm024/kdVRXnMXgDxWkqM3j+/ZVYErsbn2+/W3r/AIY1zVYoUsvFV3YsjEsyRgbh6fKRQpys3ygpzabcfyOrorzu38A+KormKR/H1+6I4ZkKN8wB6ffrX8QeF9d1VbcWXiy7sTGW3lI8b84x90jpg/nQpys3ygpzabcfyOtorzy18B+KYLyCaTx7fyxxyKzRlGwwByRy/etXxB4V17Vjbmx8W3lj5e7fsjxvzjH3SOmD+dCnK1+UFOfK3y/kddRXn1l4E8UW19bzzeO7+aOORXeMo2HAOSOXI5+lZfxB1OHTHi8z4gz2M0SkNa2yb5HPbKoRj/gVa0adas+WnBuXZai9pJRbcfyPUpv9TJ/un+VeWeK/iJH4e8OW+mafKBqBhHmy9oAR2/2j+lcVoM3xC8TzynSdT1f7Goz9qu5/LXae+0k579M9K7jwp8J7GK2ttU1DUJb++cCTM0YZFb1CtnJ9zn8K2nhZU5ezxPud0rN27aOyfrt2CSqVaV17rb+dv62OV8L6T428Y6Oun2txPpHh6Ri8t1MT5twD1C9yvXgYXk5Jr13wr4M0XwfZeRpdsBKwxLcycyy/U+nsMCro02+AwNXn/wC/Sf4Uf2dff9Bif/v0n+FdFfHynTVCjHkpr7K/V7t+pVOlGCSXQ06KzP7Ovv8AoMT/APfpP8KP7Ovv+gxP/wB+k/wrgNDTorM/s6+/6DE//fpP8KP7Ovv+gxP/AN+k/wAKANOisz+zr7/oMT/9+k/wo/s6+/6DE/8A36T/AAoA06KzP7Ovv+gxP/36T/Cj+zr7/oMT/wDfpP8ACgDTorM/s6+/6DE//fpP8KP7Ovv+gxP/AN+k/wAKANOmRf6paz/7Ovv+gxP/AN+k/wAKZFp195a/8Tef/v2n+FAGtRWZ/Z19/wBBif8A79J/hR/Z19/0GJ/+/Sf4UAadFZn9nX3/AEGJ/wDv0n+FH9nX3/QYn/79J/hQBp0Vmf2dff8AQYn/AO/Sf4Uf2dff9Bif/v0n+FAGnRWZ/Z19/wBBif8A79J/hR/Z19/0GJ/+/Sf4UAadFZn9nX3/AEGJ/wDv0n+FH9nX3/QYn/79J/hQBp0Vmf2dff8AQYn/AO/Sf4Uf2dff9Bif/v0n+FAGnTIv9X+J/nWf/Z19/wBBif8A79J/hTI9Ovtn/IXn6n/lmnr9KANaisz+zr7/AKDE/wD36T/Cj+zr7/oMT/8AfpP8KANOisz+zr7/AKDE/wD36T/Cj+zr7/oMT/8AfpP8KANOisz+zr7/AKDE/wD36T/Cj+zr7/oMT/8AfpP8KANOioreOSGBUlmaZxnLsACefapaACsyfw5odzO88+jadLK53PJJaozMfUkjmtOuA1bQ/iXPq11Lpni/T7axeQmCB7FWaNOwJ280AdZB4d0O2mWa30bTopVOVeO1RWB9iBWD8ONXv9Z0bVJtRuWnkh1a6gjZgBtjVsKvA7Vi/wDCPfFn/oeNL/8ABcn/AMTXQ+APDF94T0Cey1G8ivLqe8lupJYk2qS5BPH1zQB1VFFFABXI+JvBDaxrdtr+kavPo2uW8RgF1FEsqyRk52SRtww9Of5DHXUUAcp4W8EjQtUvda1HVJ9X1y9QRy3s0axhUHREQcKvA4yegrT8W/8AIma7/wBg+4/9FtWxWP4t/wCRM13/ALB9x/6LagDQsf8AkH23/XJf5CrFV7H/AJB9t/1yX+QqxQAUUUUAFFFFAEU8UcsTCSNXGDwwzVR9E0mdB5ul2UnH8duh/pV5/wDVt9DSr9wfSqjOUfhdgMWXwZ4Xm/1nhzSWPr9ijz+eKoTfDXwZP9/w7ZD/AHFKfyIrqqK3jjMTH4akl82LlXY4Wb4PeBpumjGM+qXUo/8AZsVnz/AzwdL9wahD/wBc7gH/ANCBr0qiuiObY+O1aX3tk+zj2PK/+FHadb/8g/xHrNt6fvFOPyC0f8Kq8S23/Hj8RtWjA6I4cj/0Z/SvVKK0/tvHP4p39VF/mg9nE8r/AOEQ+Kdn/wAeXji2mx0+0x5/mjUbPjNZf8tdGv8AHsq5/RK9Uoo/tab+OlB/9uL9LB7Nd2eV/wDCUfFiz/4+/BljOB3t5Bk/lI38qP8AhaXiq041D4c6ooHV4i5H/ovH616pRR/aGHl/Ew0fk5L9Q5H0Z5WPjnpducal4f1m0Pf92px+ZWr1v8bvBk+PMuLy3/662xP/AKDmvRiMjB6VQuNE0m8z9p0uynz1823Rv5ij6xl0vioyXpP/ADiFp9znoPin4Juh+78QW65/56I8f/oSitSz8X+GrqNBB4g0t2wPlF2mfyzmq1z8PvCF0D5nhzTh/wBc4BH/AOg4rIl+D3ge5jDf2Q0TEcmO5lH6FiKLZXLrUX/gL/yD3/I7aG7trgZguIpR/sOG/lU1eYTfAjwjIcpNqkP+5Opx+aGov+FIWcH/AB5eJtat/T94D/ICj6tl72rtesP8mw5p9j1SivK/+FRavH/x7fEHWovTl/6SCj/hVvitfu/E7WPxWT/49R9SwXTEr/wGX+Qc0v5T1SivK/8AhV3i1vvfE3Vx/urJ/wDHqP8AhUesy/8AH18Qtam/Fx/OQ0fUsF1xK/8AAZf5BzS/lPVCQBknAFZV54m0HTgftmtafBjqJLlFP5ZrgB8DNJnOdR17Wbo+8qjP5qa1LP4L+CrUgyWE90R3nuX/AJKQKPYZbD4q0pekbfmwvPsSaj8YvBenghdSe7cfwWsLN+pAX9axD8VfEGu/L4T8F3twrfdubvIT8ccf+P13uneEfDmkkNYaJYQOOkiwLv8A++iM/rW1T+s5fS/h0XJ95S/SNvzFab3Z5MfCPxI8Vf8AIxeJI9JtH+9a2H3seh24B/FmrX0f4b+EPD1z9ltoEu9YMTSRy3w87aR0YrjaOfbPvXZ3WqwW1/bWIV5bmc/6uMZKL3dvQUunaVBpxmkVnlnncvLNIcu3oM+g9KxrZtiakfZ02oR6qK5V+Gr+bOiFGEFz1N+nn/wPzIdO0mPS7Kb5zNcygtPO33pGx+gHYVJoX/IDs/8ArkKuzf6mT/dP8qpaF/yA7P8A65CvOSSVkKc5VJOUnqaFFFFMgKKKKACiiigAooooAKKKKACiiigApkX+qWn0yL/VLQA+iiigAooooAKKKKACiiigAooooAKKKKACmRf6v8T/ADp9Mi/1f4n+dAD6KKKACiiigAooooAKKKKACuSPg6aTSNQ00eLNdElzc/aBcrdfvrcdfLQ4+VPautrz34bGJtX8dXU5AvTr0ySluCIUAEWfbG7H40AZOp+EtP0WaOHVPi34hspZOUS41dY2YeuD2967/wAMaQdF0ZbY6zfauHYyrdXs3muQcYAb09PrXmvw+8L6N49/4SDxb4gsI9QfUdQljtfPyQlumAu309M9flFdJ8JA1t4e1fSVkaS10rWruxtWY5PlKwI5/wCBGgDv6KKKACiiigArH8W/8iZrv/YPuP8A0W1bFY/i3/kTNd/7B9x/6LagDQsf+Qfbf9cl/kKsVXsf+Qfbf9cl/kKsUAFFFFABRRRQA1/9W30NKv3B9KR/9W30NKv3B9KAFooooAKKKKACiiigAooooAKKKKACiiigBD90/Skj/wBUn+6KU/dP0pI/9Un+6KAHUUUUAFFFFABRRRQAUUUUAFZeq6ubKSK0tYvtGoT/AOrhzgAd2Y9lFSatfzWUCLa2zXF1O2yFADtB9WPYDrU9nBJHBE100ct2ECyTKgXd3x9Klu+iN4RUEqk1ddF3/wCB/SCCxt4Lma6SFVuJ8GV8kk4GMZParNFFUlYxlJyd2xk3+pk/3T/KqWhf8gOz/wCuQq7N/qZP90/yqloX/IDs/wDrkKBGhRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABTIv9UtPpkX+qWgB9FFFABRRRQAUUUUAFFFFABRRRQAUUUUAFMi/1f4n+dPpkX+r/E/zoAfRRRQAUUUUAFFFFABRRRQAVw/iX4VeH/E2rS6nLNqNjc3ChLk2Fx5YuFAxhwQQeOO1dxRQBQ0nR7LQtGt9J0yL7PaW8eyJVOSvvk9Tkk5Peqvhjw1ZeFNGGmWMk8sfmvM8twwaSR3bJLEADPbp2rZooAKKKKACiiigArH8W/8AIma7/wBg+4/9FtWxWP4t/wCRM13/ALB9x/6LagDQsf8AkH23/XJf5CrFV7H/AJB9t/1yX+QqxQAUUUUAFFFFADX/ANW30NKv3B9KR/8AVt9DSr9wfSgBaKKKACiiigAooooAKKKKACiiigAooooAQ/dP0pI/9Un+6KU/dP0pI/8AVJ/uigB1FFFABRRRQAUUUUAFVdQnuILOVrOAXFyANke4DknGT7f4UtxdxxyrbJLELuVGMMbn7xA/lVTSNKax8y5upfPv7jBml7eyqOyipbvojaEVFc8/ku//AACfS7S4s7Tbd3T3NxIxeRyeAT2UdgKu0UU0rKxnObnJyfUKKKKZIyb/AFMn+6f5VS0L/kB2f/XIVdm/1Mn+6f5VS0L/AJAdn/1yFAGhRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABTIv9UtPpkX+qWgB9FFFABRRRQAUUUUAFFFFABRRRQAUUUUAFMi/1f4n+dPpkX+r/ABP86AH0UUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABWP4t/5EzXf+wfcf+i2rYrH8W/8iZrv/YPuP/RbUAaFj/yD7b/rkv8AIVYqvY/8g+2/65L/ACFWKACiiigAooooAa/+rb6GlX7g+lI/+rb6GlX7g+lAC0UUUAFFFFABRRRQAUUUUAFFFFABRRRQAh+6fpSR/wCqT/dFKfun6Ukf+qT/AHRQA6iiigAooooAKp3mp21jPbwSFmmuHCRxoNzH1OPQdzUWratHpkaKsZnu5jtgt0PzSH+gHc1LBp0CX0moGIi7mRVYs27aAPur6D6VLd9EbQpqK56i0d7ef/A8/kRWGkx2d3cXkkrXF3OxzK45VM8KPQD9a0aKKaSWiInOU3eTCiiimQFFFFADJv8AUyf7p/lVLQv+QHZ/9chV2b/Uyf7p/lVLQv8AkB2f/XIUAaFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFMi/1S0+mRf6paAH0UUUAFFFFABRRRQAUUUUAFFFFABRRRQAUyL/V/if50+mRf6v8AE/zoAfRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFY/i3/kTNd/7B9x/6Latisfxb/yJmu/9g+4/9FtQBoWP/IPtv+uS/wAhViq9j/yD7b/rkv8AIVYoAKKKKACiiigBr/6tvoaVfuD6Uj/6tvoaVfuD6UALRRRQAUUUUAFFFFABRRRQAUUUUAFFFFACH7p+lJH/AKpP90Up+6fpSR/6pP8AdFADqKKKACqGq6mumWquInmnkby4YUHLueg9h71PfXElrZSzRW8lxIo+WKPqxqDSoL6K2Z9RnEtxK28oo+WL/ZX2FS29kbU4xS9pLVX2vuOsraQxwXN/HAdQEZVpI16AnO0HrirtFFNKxnKTk7sKKKKZIUUUUAFFFFADJv8AUyf7p/lVLQv+QHZ/9chV2b/Uyf7p/lVLQv8AkB2f/XIUAaFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFMi/1S0+mRf6paAH0UUUAFFFFABRRRQAUUUUAFFFFABRRRQAUyL/V/if50+mRf6v8AE/zoAfRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFY/i3/kTNd/7B9x/6Latisfxb/yJmu/9g+4/9FtQBoWP/IPtv+uS/wAhViq9j/yD7b/rkv8AIVYoAKKKKACiiigBr/6tvoaVfuD6Uj/6tvoaVfuD6UALRRRQAUUUUAFFFFABRRRQAUUUUAFFFFACH7p+lJH/AKpP90Up+6fpSR/6pP8AdFADqimnSIrHvjE0gPlIzY3kDOBUV7qFtYCL7RJtMriONQCSzH0A5qtaaR5epzajdzfaLliViJXAhj7Ko9fU1LetkawguXmnounm/wCtxukafcwyS3+oS776cAMqn5IlHRFH9f8AJ1aKKaSSsialR1JczCiiimQFFFFABRRRQAUUUUAMm/1Mn+6f5VS0L/kB2f8A1yFXZv8AUyf7p/lVLQv+QHZ/9chQBoUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUyL/VLT6ZF/qloAfRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABTIv9X+J/nT6ZF/q/xP86AH0UUUAFFFFABRRRQAUUUUAFFFFABRRRQAUVXv7630zTrm/vJPLtbWJ5pn2k7UUEscDk4APSuH/wCF2/Dz/oYf/JK4/wDjdAHoFcdrfj7+zNan0jT/AA3rer3VuFMzWluPKTcoYAuTjOCOKXRPil4N8R6xBpOk6z9ovp93lxfZZk3bVLHlkAHAJ5NdTeXltp9nNeXk6QW0KF5JZDhVUdSTQBgeE/G+neLHu7eGC7sdRsiBdWN7F5csWehx3H+eMir3i3/kTNd/7B9x/wCi2ri/AUU/iTx1rXj77O9rpt3Atlp6yLteeNSCZSPQlRj2+nPS+Ory9t/C+qxW+ly3UUlhOJJklRRF8h5IYgn14oA37H/kH23/AFyX+QqxVSxd/wCz7b92f9Uvcegqxvf/AJ5n8xQA+imb3/55n8xRvf8A55n8xQA+imb3/wCeZ/MUb3/55n8xQAr/AOrb6GlX7g+lRu77G/dnoe4pVd9g/dnp6igCSimb3/55n8xRvf8A55n8xQA+imb3/wCeZ/MUb3/55n8xQA+imb3/AOeZ/MUb3/55n8xQA+imb3/55n8xRvf/AJ5n8xQA+imb3/55n8xRvf8A55n8xQA+imb3/wCeZ/MUb3/55n8xQA4/dP0qjf6pBpVgk02WdsLFEnLSMeigUmp6pHpdmZ542OTtRFwWdj0UD1qKwgkuYrS+1CxiS+jQhdrZ8sHt9cY/Wpb6Lc3p00kqlRe7+f8AXV9CWLTYZNRGqzJJ9paIKqSMCIeOQuOAfU1oUze//PM/mKN7/wDPM/mKaSRlOcp7j6KZvf8A55n8xRvf/nmfzFMkfRTN7/8APM/mKN7/APPM/mKAH0Uze/8AzzP5ije//PM/mKAH0Uze/wDzzP5ije//ADzP5igB9FM3v/zzP5ije/8AzzP5igAm/wBTJ/un+VUtC/5Adn/1yFWpnfyZP3Z+6e4qlobP/Ydn+7P+rHcUAadFM3v/AM8z+Yo3v/zzP5igB9FM3v8A88z+Yo3v/wA8z+YoAfRTN7/88z+Yo3v/AM8z+YoAfRTN7/8APM/mKN7/APPM/mKAH0Uze/8AzzP5ije//PM/mKAH0Uze/wDzzP5ije//ADzP5igB9Mi/1S0b3/55n8xTInfyl/dn8xQBNRTN7/8APM/mKN7/APPM/mKAH0Uze/8AzzP5ije//PM/mKAH0Uze/wDzzP5ije//ADzP5igB9FM3v/zzP5ije/8AzzP5igB9FM3v/wA8z+Yo3v8A88z+YoAfRTN7/wDPM/mKN7/88z+YoAfTIv8AV/if50b3/wCeZ/MUyJ32f6s9T3HrQBNRTN7/APPM/mKN7/8APM/mKAH0Uze//PM/mKN7/wDPM/mKAH0Uze//ADzP5ije/wDzzP5igB9FICSMkYPpS0AFFFFABRRRQAUUUUAFeRfEK28W654visn8MXWpeE7MpJ5FvdRxC9k2g/vCxztUkjbjnGe4x67RQByvhjXtd1G8NnqPg6fRLSOHMcz3Uci5BACBV6cEn04rS8W/8iZrv/YPuP8A0W1bFY/i3/kTNd/7B9x/6LagDQsf+Qfbf9cl/kKsVXsf+Qfbf9cl/kKsUAFFFFABRRRQA1/9W30NKv3B9KR/9W30NKv3B9KAFooooAKKKKACiiigAooooAKKKKACoLy5FnZy3BjkkEa7tka7mb2AqV3CYBI3Nwqk43HGcCsvSrW/e4l1DUpGSWUbY7VHykKZ79i3vUt9Ea04KzlJ6Lp3/ruSaYl/LbvPqewSSPvjgCg+QMcDPc+prQj/ANUn+6KU/dP0pI/9Un+6KaVlYmc+eV7W9B1FFFMgKKKKACiiigAooooAKKKKACiiigBk3+pk/wB0/wAqpaF/yA7P/rkKuzf6mT/dP8qpaF/yA7P/AK5CgDQooooAKKKKACiiigAooooAKKKKACiiigApkX+qWn0yL/VLQA+iiigAooooAKKKKACiiigAooooAKKKKACmRf6v8T/On0yL/V/if50APooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACsfxb/wAiZrv/AGD7j/0W1bFY/i3/AJEzXf8AsH3H/otqANCx/wCQfbf9cl/kKsVXsf8AkH23/XJf5CrFABRRRQAUUUUANf8A1bfQ0q/cH0pH/wBW30NKv3B9KAFooooAKKKKACiiigAooooAKgu722sYRLdTLEhYKC3cnoKbf39vplm91cvtjX82PYAdzVRNOivr+DVbjziVjBht5QAIWPU4/vdPp/KW+i3NqdNW56nw/m+39bCRaTJJrD6jfyrM0ZK2sajCxL647sfWtWiimklsROpKdubpoIfun6Ukf+qT/dFKfun6Ukf+qT/dFMgdRRRQAUUUUAFFFFABRRRQAUUUUAFFFFADJv8AUyf7p/lVLQv+QHZ/9chV2b/Uyf7p/lVLQv8AkB2f/XIUAaFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFMi/1S0+mRf6paAH0UUUAFFFFABRRRQAUUUUAFFFFABRRRQAUyL/V/if50+mRf6v8AE/zoAfRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFY/i3/kTNd/7B9x/6Latisfxb/yJmu/9g+4/9FtQBoWP/IPtv+uS/wAhViq9j/yD7b/rkv8AIVYoAKKKKACiiigBr/6tvoaVfuD6Uj/6tvoaVfuD6UALRRRQAUUUUAFFFFABVa+vrfTbN7q6kCRIPxJ7ADuakuriO0tZbiUkRxKWbAycD2rP003WpQtc6lbRJEziS2gdMvGB0LH+939qlvotzanTuueXwr8fJC2sA1VLTUNQsTDPEWaGJ3J2A9CR03Yx9K1KKKaViJzc35dF2CiiimQIfun6Ukf+qT/dFKfun6Ukf+qT/dFADqKKKACiiigAooooAKKKKACiiigAooooAZN/qZP90/yqloX/ACA7P/rkKuzf6mT/AHT/ACqloX/IDs/+uQoA0KKKKACiiigAooooAKKKKACiiigAooooAKZF/qlp9Mi/1S0APooooAKKKKACiiigAooooAKKKKACiiigApkX+r/E/wA6fTIv9X+J/nQA+iiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKx/Fv/Ima7/2D7j/0W1bFY/i3/kTNd/7B9x/6LagDQsf+Qfbf9cl/kKsVXsf+Qfbf9cl/kKsUAFFFFABRRRQA1/8AVt9DSr9wfSkf/Vt9DSr9wfSgBaKKKACiiigAoJxUVxcQ2sLTXEqRRL1dzgCs0abc3es/bL6VTb27f6JBGxxnH329T7dqlvsaQgmm5Oy/PyQaYupXV5Jf3pe3iIKQ2eRwM/ef/aOPwrXooppWQVKnPK9regUUUUzMKKKKAEP3T9KSP/VJ/uilP3T9KSP/AFSf7ooAdRRRQAUUUUAFFFFABRRRQAUUUUAFFFFADJv9TJ/un+VUtC/5Adn/ANchV2b/AFMn+6f5VS0L/kB2f/XIUAaFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFMi/1S0+mRf6paAH0UUUAFFFFABRRRQAUUUUAFFFFABRRRQAUyL/V/if50+mRf6v8T/OgB9FFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAVj+Lf+RM13/sH3H/AKLatisfxb/yJmu/9g+4/wDRbUAaFj/yD7b/AK5L/IVYqvY/8g+2/wCuS/yFWKACiiigAooooAa/+rb6GlX7g+lI/wDq2+hpV+4PpQAtFFFABUN1dQ2VtJcXEixxRjLM3alubmG0t5Li4kWOKMZZm6AVmQxQeIre2vLq1mjijkLxRSNxIP4WZf1ANS30W5rTpp+/P4f60/rYVrCLWbm01CaSR7RUEkVq6bQHP8TDucdBWvRRTSsKdRzsui2CiiimZhRRRQAUUUUAIfun6Ukf+qT/AHRSn7p+lJH/AKpP90UAOooooAKKKKACiiigAooooAKKKKACiiigBk3+pk/3T/KqWhf8gOz/AOuQq7N/qZP90/yqloX/ACA7P/rkKANCiiigAooooAKKKKACiiigAooooAKKKKACmRf6pafTIv8AVLQA+iiigAooooAKKKKACiiigAooooAKKKKACmRf6v8AE/zp9Mi/1f4n+dAD6KKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigArH8W/8iZrv/YPuP8A0W1bFY/i3/kTNd/7B9x/6LagDQsf+Qfbf9cl/kKsVXsf+Qfbf9cl/kKsUAFFFFABRRRQA1/9W30NKv3B9KR/9W30NKv3B9KAFpk00dvBJNM4SONSzMewFPrIsJdQ1DUJbmZGtrBA0cdu6/NKc8u2eg44FJu2hpTp8ycm9F/VkNsZH16GSW9soxYM6tbJKPnYDnew7A8YFbNFFCVgqT53orLov6/EKKKKZmFFFFABRRRQAUUUUAIfun6Ukf8Aqk/3RSn7p+lJH/qk/wB0UAOooooAKKKKACiiigAooooAKKKKACiiigBk3+pk/wB0/wAqpaF/yA7P/rkKuzf6mT/dP8qpaF/yA7P/AK5CgDQooooAKKKKACiiigAooooAKKKKACiiigApkX+qWn0yL/VLQA+iiigAooooAKKKKACiiigAooooAKKKKACmRf6v8T/On0yL/V/if50APooooAKKKKACiiigAooooAKKKKACiiigAooooAKKK47xzrV4otPDGhybdb1glEkHP2WAf6yY+mBwPUnjpQB1Vte2l5v+y3UM/lna/lSBtp9DjpWd4t/5EzXf+wfcf+i2rgPg5plvouseNtMtAwt7TUlhj3HJwoYcn1rtPG+qadY+FNXt7u/tbeaawnEUcsyo0h2EfKCcnn0oA27H/kH23/XJf5CrFVbGRP7PtvnX/VL39hVjzE/vr+dADqKb5if31/OjzE/vr+dADqKb5if31/OjzE/vr+dAA/8Aq2+hpGkSKHzJHVEVclmOAB9aZPcQxW8kkkqKiqSzFgABWRLbQ+IDZztdq+lhBJ5ABXzXzxuz/CPSpbttua06ak7zdo9/09SZ7O+vtZWW4k8mwtmDQxxvzM2PvMR2HTFa9MDxgABlAHQA0vmJ/fX86aVhTqOdl0Q6im+Yn99fzo8xP76/nTMx1FN8xP76/nR5if31/OgB1FN8xP76/nR5if31/OgB1FN8xP76/nR5if31/OgB1FN8xP76/nR5if31/OgBT90/Skj/ANUn+6KQyJtPzr+dJHInlJ869B3oAkopvmJ/fX86PMT++v50AOopvmJ/fX86PMT++v50AOopvmJ/fX86PMT++v50AOopvmJ/fX86PMT++v50AOopvmJ/fX86PMT++v50AOopvmJ/fX86PMT++v50AJN/qZP90/yqloX/ACA7P/rkKtzSJ5Mnzr9096paHIn9h2fzr/qh3oA0qKb5if31/OjzE/vr+dADqKb5if31/OjzE/vr+dADqKb5if31/OjzE/vr+dADqKb5if31/OjzE/vr+dADqKb5if31/OjzE/vr+dADqKb5if31/OjzE/vr+dADqZF/qlpfMT++v50yKRPKX51/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KACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA4rxzrWueFbix8QWzG60CE+XqlmsSl0QniZDjPy55GcdOnJFfw94n1Pxp4vnutHuBF4S08GEzeUCb+fvtJGQi+oxn8eDx7Jq/iC9tvBejxXMEN8u/U9S8shILbuitjBdumPQ++RU8J2V/wDD/wATt4U8i6ufDV6Wn0y6CF/sr9XhkIHAJOQT6+5wAekUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAH//2Q==)  
图 3：比较 SiLU（又名 Swish）和 ReLU 激活函数。

在原始的 Transformer 论文（Vaswani 等人 [2017] 的第 3.3 节）中，Transformer 前馈网络由两个线性变换组成，中间有一个 ReLU 激活函数 $(\mathrm{ReLU}(x) = \max(0, x))$。内部前馈层的维度通常是输入维度的 $4\mathrm{x}$。

然而，与原始设计相比，现代语言模型倾向于包含两个主要变化：它们使用另一种激活函数并采用门控机制。具体来说，我们将实现 LLM（如 Llama 3 [Grattafori et al., 2024] 和 Qwen 2.5 [Yang et al., 2024]）中采用的“SwiGLU”激活函数，它将 SiLU（通常称为 Swish）激活与称为门控线性单元（GLU）的门控机制相结合。我们还将省略有时在线性层中使用的偏置项，遵循自 PaLM [Chowdhery et al., 2022] 和 LLaMA [Touvron et al., 2023] 以来大多数现代 LLM 的做法。

SiLU 或 Swish 激活函数 [Hendrycks and Gimpel, 2016, Elfwing et al., 2017] 定义如下：

$$
\operatorname {S i L U} (x) = x \cdot \sigma (x) = \frac {x}{1 + e ^ {- x}} \tag {5}
$$

如图 3 所示，SiLU 激活函数与 ReLU 激活函数类似，但在零点处是平滑的。

门控线性单元（GLU）最初由 Dauphin 等人 [2017] 定义，作为通过 sigmoid 函数的线性变换与另一个线性变换的逐元素乘积：

$$
\operatorname {G L U} (x, W _ {1}, W _ {2}) = \sigma \left(W _ {1} x\right) \odot W _ {2} x, \tag {6}
$$

其中 $\odot$ 表示逐元素乘法。门控线性单元被建议“通过为梯度提供线性路径同时保留非线性能力来减少深度架构的梯度消失问题”。

将 SiLU/Swish 和 GLU 结合起来，我们得到 SwiGLU，我们将将其用于我们的前馈网络：

$$
\operatorname {F F N} (x) = \operatorname {S w i G L U} (x, W _ {1}, W _ {2}, W _ {3}) = W _ {2} (\operatorname {S i L U} (W _ {1} x) \odot W _ {3} x), \tag {7}
$$

其中 $x\in \mathbb{R}^{d_{\mathrm{model}}}$ ， $W_{1},W_{3}\in \mathbb{R}^{d_{\mathrm{ff}}\times d_{\mathrm{model}}}$ ， $W_{2}\in \mathbb{R}^{d_{\mathrm{model}}\times d_{\mathrm{ff}}}$ ，并且通常情况下，$d_{\mathrm{ff}} = \frac{8}{3} d_{\mathrm{model}}$ 。

Shazeer [2020] 首先提出了将 SiLU/Swish 激活与 GLUs 结合，并通过实验表明 SwiGLU 在语言建模任务上优于 ReLU 和 SiLU（无门控）等基线模型。稍后，你将在作业中比较 SwiGLU 和 SiLU。尽管我们已经提到了一些关于这些组件的启发式论证（并且论文提供了更多支持性证据），但保持经验性的视角仍然很重要：Shazeer 论文中的一句名言是：

我们不解释为什么这些架构似乎有效；我们将它们的成功，以及其他一切，都归因于神圣的仁慈。

# 问题 (positionwise_feedforward): 实现位置前馈网络（2 分）

Deliverable: 实现 SwiGLU 前馈网络，该网络由 SiLU 激活函数和 GLU 组成。

Note: 在这种特殊情况下，为了数值稳定性，您可以随意在实现中使用 torch.sigmoid。

在您的实现中，您应该将 $d_{\mathrm{ff}}$ 设置为大约 $\frac{8}{3} \times d_{\mathrm{model}}$，同时确保内部前馈层的维度是 64 的倍数，以充分利用您的硬件。要使用我们提供的测试来测试您的实现，您需要实现 [adapters.run_swiglu] 中的测试适配器。然后，运行 uv run pytest -k test_swiglu 来测试您的实现。

# 3.5.3 相对位置嵌入

为了将位置信息注入模型，我们将实现旋转位置嵌入 [Su et al., 2021]，通常称为 RoPE。对于给定查询词元 $q^{(i)} = W_q x^{(i)} \in \mathbb{R}^d$ 在词元位置 $i$，我们将应用一个成对旋转矩阵 $R^i$，得到 $q'^{(i)} = R^i q^{(i)} = R^i W_q x^{(i)}$。这里，$R^i$ 将通过角度 $\theta_{i,k} = \frac{i}{\Theta(2k-2)/d}$ 将嵌入元素的对 $q_{2k-1:2k}^{(i)}$ 作为二维向量旋转，其中 $k \in \{1, \dots, d/2\}$ 且 $\Theta$ 为某个常数。因此，我们可以将 $R^i$ 视为一个大小为 $d \times d$ 的块对角矩阵，其块为 $R_k^i$，其中 $k \in \{1, \dots, d/2\}$，

$$
R _ {k} ^ {i} = \left[ \begin{array}{c c} \cos (\theta_ {i, k}) & - \sin (\theta_ {i, k}) \\ \sin (\theta_ {i, k}) & \cos (\theta_ {i, k}) \end{array} \right]. \tag {8}
$$

因此，我们得到完整的旋转矩阵

$$
$ R ^ {i} = \left[ \begin{array}{c c c c c} R _ {1} ^ {i} & 0 & 0 & \dots & 0 \\ 0 & R _ {2} ^ {i} & 0 & \dots & 0 \\ 0 & 0 & R _ {3} ^ {i} & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \dots & R _ {d / 2} ^ {i} \end{array} \right], \tag {9} $

其中 0s 表示 $2 \times 2$ 的零矩阵。虽然可以构造完整的 $d \times d$ 矩阵，但一个好的解决方案应该利用该矩阵的性质来更有效地实现变换。由于我们只关心给定序列中词元之间的相对旋转，因此我们可以跨层和不同的批次重用我们为 $\cos(\theta_{i,k})$ 和 $\sin(\theta_{i,k})$ 计算出的值。如果您想对其进行优化，可以使用一个由所有层引用的 RoPE 模块，并且它可以在初始化时使用 self.register_buffer(persistent=False) 创建一个 2d 预计算缓冲区（包含 sin 和 cos 值），而不是使用 nn_PARAMETER（因为我们不想学习这些固定的余弦和正弦值）。然后，我们对 $k^{(j)}$ 执行与对 $q^{(i)}$ 执行的完全相同的旋转过程，通过相应的 $R^j$ 进行旋转。请注意，此层没有可学习的参数。

Deliverable: 实现一个 RotaryPositionalEmbedding 类，将 RoPE 应用于输入张量。推荐使用以下接口：

Problem (rope): 实现 RoPE (2 分)
```python
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)
```
构造 RoPE 模块并根据需要创建缓冲区。

theta: float RoPE 的 $\Theta$ 值

d_k: int 查询和键向量的维度

max_seq_len: int 将输入的序列的最大长度

device: torch_device | None = None 用于存储缓冲区的设备

Problem (softmax): 实现 softmax (1 分)
```python
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
```
处理形状为 (…, seq_len, d_k) 的输入张量，并返回相同形状的张量。

请注意，您应该能够处理具有任意数量批处理维度的 $x$。您应该假设 token 位置是一个形状为 (..., seq_len) 的张量，指定 $x$ 沿序列维度的 token 位置。

您应该使用 token 位置沿序列维度对（可能预先计算的）cos 和 sin 张量进行切片。

要测试您的实现，请完成 [adapters.run_ripe] 并确保它通过 uv run.pytest -k test_ripe。

# 3.5.4 缩放点积注意力

现在我们将实现 Vaswani et al. [2017]（第 3.2.1 节）中描述的缩放点积注意力。作为初步步骤，注意力操作的定义将使用 softmax，这是一种将未归一化的分数向量转换为归一化分布的操作：

$$
\operatorname {s o f t m a x} (v) _ {i} = \frac {\exp \left(v _ {i}\right)}{\sum_ {j = 1} ^ {n} \exp \left(v _ {j}\right)}. \tag {10}
$$

请注意，对于较大的值，$\exp(v_i)$ 可能会变成 inf（然后，$\inf / \inf = \mathsf{NaN}$）。我们可以通过注意到 softmax 操作对于向所有输入添加任何常数 $c$ 都是不变的来避免这种情况。我们可以利用此属性来提高数值稳定性——通常，我们会从 $o_i$ 的所有元素中减去 $o_i$ 的最大条目，使新的最大条目为 0。您现在将实现 softmax，使用此技巧来提高数值稳定性。

交付物：编写一个函数来对张量应用 softmax 操作。您的函数应接受两个参数：一个张量和一个维度 $i$，并将 softmax 应用于输入张量的第 $i$ 维。输出张量应与输入张量具有相同的形状，但其第 $i$ 维现在将具有归一化的概率分布。使用从第 $i$ 维的所有元素中减去第 $i$ 维的最大值这一技巧来避免数值稳定性问题。

To test your implementation, complete [adapters.runsoftmax] and make sure it passes uv run pytest -k testSoftmax_MATCHes_pytorch.

We can now define the Attention operation mathematically as follows:

$$
\operatorname {A t t e n t i o n} (Q, K, V) = \operatorname {s o f t m a x} \left(\frac {Q ^ {\top} K}{\sqrt {d _ {k}}}\right) V \tag {11}
$$

where $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{m \times d_k}$, and $V \in \mathbb{R}^{m \times d_v}$. Here, $Q$, $K$ and $V$ are all inputs to this operation—note that these are not the learnable parameters. If you're wondering why this isn't $QK^{\top}$, see 3.3.1.

掩码：有时将注意力操作的输出进行掩码会很方便。掩码的形状应为 $M \in \{\text{True}, \text{False}\}^{n \times m}$，此布尔矩阵的每一行 $i$ 表示查询 $i$ 应关注哪些键。按照惯例（并且有点令人困惑的是），位置 $(i, j)$ 的值为 True 表示查询 $i$ 关注键 $j$，值为 False 表示查询 $i$ 不关注键 $j$。换句话说，“信息流”在值为 True 的 $(i, j)$ 对处流动。例如，考虑一个条目为 [[True, True, False]] 的 $1 \times 3$ 掩码矩阵。单个查询向量仅关注前两个键。

在计算上，使用掩码比在子序列上计算注意力要高效得多，我们可以通过取预 softmax 值 $\left(\frac{Q^\top K}{\sqrt{d_k}}\right)$ 并将掩码矩阵中为 False 的任何条目加上 $-\infty$ 来实现这一点。

# 问题 (scaled.dot_productattention): 实现缩放点积注意力 (5 分)

交付物：实现缩放点积注意力函数。你的实现应该能够处理形状为 (batch_size, ..., seq_len, d_k) 的键和查询，以及形状为 (batch_size, ..., seq_len, d_v) 的值，其中 ... 代表任何数量的其他类似批次的维度（如果提供）。实现应该返回形状为 (batch_size, ..., d_v) 的输出。有关类似批次的维度的讨论，请参阅第 3.3 节。

你的实现还应该支持一个可选的用户提供的布尔掩码，形状为 (seq_len, seq_len)。掩码值为 True 的位置的注意力概率应该加起来等于 1，而掩码值为 False 的位置的注意力概率应该为零。要针对我们提供的测试来测试你的实现，你需要实现 [adapters.runScaled.dot_productattention] 中的测试适配器。

uv run pytest -k testScaled.dot_productattention 测试您在三阶输入张量上的实现，而 uv run pytest -k test_4dScaled.dot_productattention 测试您在四阶输入张量上的实现。

# 3.5.5 因果多头自注意力

我们将按照 Vaswani et al. [2017] 的 3.2.2 节中的描述来实现多头自注意力。回想一下，在数学上，应用多头注意力的操作定义如下：

$$
\operatorname {M u l t i H e a d} (Q, K, V) = \operatorname {C o n c a t} \left(\operatorname {h e a d} _ {1}, \dots , \operatorname {h e a d} _ {h}\right) \tag {12}
$$

$$
\text {f o r} \quad \text {h e a d} _ {i} = \text {A t t e n t i o n} \left(Q _ {i}, K _ {i}, V _ {i}\right) \tag {13}
$$

其中 $Q_{i}$、 $K_{i}$、 $V_{i}$ 是切片号为 $i \in \{1, \dots, h\}$ 的 $Q$、 $K$ 和 $V$ 的大小分别为 $d_{k}$ 或 $d_{v}$ 的嵌入维度。Attention 是 §3.5.4 中定义的缩放点积注意力操作。由此，我们可以形成多头自注意力操作：

$$
\operatorname {M u l t i H e a d S e l f A t t e n t i o n} (x) = W _ {O} \operatorname {M u l t i H e a d} \left(W _ {Q} x, W _ {K} x, W _ {V} x\right) \tag {14}
$$

Here, the learnable parameters are $W_{Q} \in \mathbb{R}^{hd_{k} \times d_{\mathrm{model}}}$, $W_{K} \in \mathbb{R}^{hd_{k} \times d_{\mathrm{model}}}$, $W_{V} \in \mathbb{R}^{hd_{v} \times d_{\mathrm{model}}}$, and $W_{O} \in \mathbb{R}^{d_{\mathrm{model}} \times hd_{v}}$。由于在多头注意力操作中对 Q、K 和 V 进行了切片，我们可以认为 $W_{Q}$、$W_{K}$ 和 $W_{V}$ 在输出维度上是为每个头分开的。当您完成此操作后，您应该在总共三次矩阵乘法中计算键、值和查询投影。

因果掩码。你的实现应该阻止模型关注序列中的未来词元。换句话说，如果模型接收到一个词元序列 $t_1, \ldots, t_n$，并且我们想要计算前缀 $t_1, \ldots, t_i$（其中 $i < n$）的下一个词预测，模型不应该能够访问（关注）位置 $t_{i+1}, \ldots, t_n$ 处的词元表示，因为在推理过程中生成文本时，它将无法访问这些词元（并且这些未来的词元会泄露真实下一个词的身份信息，从而使语言建模预训练目标变得微不足道）。对于一个输入词元序列 $t_1, \ldots, t_n$，我们可以通过运行 $n$ 次多头自注意力（针对序列中的 $n$ 个唯一前缀）来粗略地阻止访问未来的词元。相反，我们将使用因果注意力掩码，它允许词元 $i$ 关注序列中的所有位置 $j \leq i$。你可以使用 torch.triu 或广播的 index 比较来构建此掩码，并且您应该利用 §3.5.4 中提供的缩放点积注意力实现已支持注意力掩码这一事实。

应用 RoPE。RoPE 应应用于查询和键向量，但不应用于值向量。此外，头维度应被视为批次维度，因为在多头注意力中，注意力是为每个头独立应用的。这意味着应将完全相同的 RoPE 旋转应用于每个头的查询和键向量。

问题 (multihead_self_attention)：实现因果多头自注意力（5 分）

交付成果：实现因果多头自注意力作为 torch.nnModule。您的实现应接受（至少）以下参数：

d_model：int Transformer 块输入的维度。

num_heads：int 在多头自注意力中使用的头数。

Following Vaswani et al. [2017], set $d_{k} = d_{v} = d_{\mathrm{model}} / h$ . To test your implementation against our provided tests, implement the test adapter at [adapters.run-multihead_self_attention]. Then, run uv run pytest -k test-multihead_self_attention to test your implementation.

# 3.6 The Full Transformer LM

Let's begin by assembling the Transformer block (it will be helpful to refer back to Figure 2). A Transformer block contains two 'sublayers', one for the multihead self attention, and another for the feed-forward network. In each sublayer, we first perform RMSNorm, then the main operation (MHA/FF), finally adding in the residual connection.

To be concrete, the first half (the first 'sub-layer') of the Transformer block should be implementing the following set of updates to produce an output $y$ from an input $x$,

$$
y = x + \operatorname{MultiHeadSelfAttention}(\operatorname{RMSNorm}(x)). \tag{15}
$$

问题 (transformer_block): 实现 Transformer 块 (3 分)

实现 §3.5 中描述并如图 2 所示的 Pre-norm Transformer 块。您的 Transformer 块应接受（至少）以下参数。

d_model: int Transformer 块输入的维度。

num_heads: int 多头自注意力中使用的头数。

d_ff: int 位置前馈内层的维度。

为了测试您的实现，请实现适配器 [adapters.run_transformer_block]。然后运行 uv run pytest -k test_transformer_block 来测试您的实现。

交付物：通过所提供测试的 Transformer 块代码。

现在我们将这些块组合在一起，遵循图 1 中的高层图。遵循我们关于嵌入的描述（第 3.1.1 节），将此输入到 num_layers 个 Transformer 块中，然后将其传递到三个输出层以获得词汇表上的分布。

# 问题 (transformer_lm): 实现 Transformer LM (3 分)

是时候将所有内容整合在一起了！实现 §3.1 中描述并如图 1 所示的 Transformer 语言模型。至少，您的实现应接受 Transformer 块的所有上述构造参数，以及这些附加参数：

vocab_size: int 词汇表的大小，用于确定词元嵌入矩阵的维度。

context_length: int 最大上下文长度，用于确定位置嵌入矩阵的维度。

num_layers: int 要使用的 Transformer 块的数量。

要针对我们提供的测试来测试您的实现，您首先需要实现 [adapters.run_transformer_lm] 中的测试适配器。然后，运行 uv run pytest -k test_transformer_lm 来测试您的实现。

交付物：一个通过上述测试的 Transformer LM 模块。

资源核算。能够了解 Transformer 的各个部分如何消耗计算和内存非常有用。我们将逐步进行一些基本的“FLOPs 核算”。Transformer 中绝大多数 FLOPs 是矩阵乘法，因此我们的核心方法很简单：

1. 写下 Transformer 前向传播中的所有矩阵乘法。
2. 将每个矩阵乘法转换为所需的 FLOPs。

对于第二步，以下事实将很有用：

规则：给定 $A \in \mathbb{R}^{m \times n}$ 和 $B \in \mathbb{R}^{n \times p}$，矩阵-矩阵乘积 $AB$ 需要 2mnp FLOPs。

要理解这一点，请注意 $(AB)[i,j] = A[i,:]\cdot B(:,j]$，并且该点积需要 $n$ 次加法和 $n$ 次乘法（2n FLOPs）。然后，由于矩阵-矩阵乘积 $AB$ 有 $m \times p$ 个条目，FLOPs 的总数为 $(2n)(mp) = 2mnp$。

现在，在进行下一个问题之前，最好逐一检查 Transformer 块和 Transformer LM 的每个组件，并列出所有的矩阵乘法及其相关的 FLOPs 成本。

# 问题 (transformer_accounting): Transformer LM 资源核算 (5 分)

(a) 考虑 GPT-2 XL，其配置如下：

```txt
vocab_size : 50,257  
context_length : 1,024  
num_layers : 48  
d_model : 1,600
```

num_heads : 25

d_ff:6,400

假设我们使用此配置构建模型。我们的模型将有多少可训练参数？假设每个参数都使用单精度浮点数表示，仅加载此模型需要多少内存？

交付成果：一到两句话的回答。

(b) 确定完成我们 GPT-2 XL 模型前向传播所需的矩阵乘法。这些矩阵乘法总共需要多少 FLOPs？假设我们的输入序列有 context_length 个词元。

<output>
交付成果：矩阵乘法列表（附说明）以及所需的总 FLOPs。

(c) 根据您上面的分析，模型的哪些部分需要最多的 FLOPs？

交付成果：一到两句话的回答。

(d) 使用 GPT-2 small（12 层，768 d_model，12 头）、GPT-2 medium（24 层，1024 d_model，16 头）和 GPT-2 large（36 层，1280 d_model，20 头）重复您的分析。随着模型尺寸的增加，Transformer LM 的哪些部分占总 FLOPs 的比例会增加或减少？

交付成果：对于每个模型，提供模型组件及其相关 FLOPs 的明细（作为前向传播所需总 FLOPs 的比例）。此外，提供一到两句话的描述，说明改变模型尺寸如何改变每个组件的比例 FLOPs。
</output>

(e) 将 GPT-2 XL 的上下文长度增加到 16,384。一次前向传播的总 FLOPs 如何变化？模型组件的 FLOPs 的相对贡献如何变化？

交付物：一到两句话的回答。

# 4 训练 Transformer LM

我们现在有了预处理数据（通过分词器）和模型（Transformer）的步骤。剩下的就是构建支持训练的所有代码。这包括：

- 损失：我们需要定义损失函数（交叉熵）。
- 优化器：我们需要定义优化器来最小化此损失（AdamW）。
- 训练循环：我们需要所有支持性基础设施来加载数据、保存检查点和管理训练。

# 4.1 交叉熵损失

我们回顾一下 Transformer 语言模型为长度为 $m + 1$ 的每个序列 $x$ 和 $i = 1, \ldots, m$ 定义了一个分布 $p_{\theta}(x_{i + 1} \mid x_{1:i})$。给定一个由长度为 $m$ 的序列组成的训练集 $D$，我们定义了标准的交叉熵（负对数似然）损失函数：

$$
\ell (\theta ; D) = \frac {1}{| D | m} \sum_ {x \in D} \sum_ {i = 1} ^ {m} - \log p _ {\theta} \left(x _ {i + 1} \mid x_{1: i}\right). \tag {16}
$$

（请注意，Transformer 中的一次前向传播会为所有 $i = 1, \dots, m$ 产生 $p_{\theta}(x_{i + 1} \mid x_{1:i})$。）

特别是，Transformer 为每个位置 $i$ 计算 logits $o_i \in \mathbb{R}^{\text{vocab-size}}$，这导致：

$$

p \left(x _ {i + 1} \mid x _ {1: i}\right) = \operatorname {softmax} \left(o _ {i}\right) \left[ x _ {i + 1} \right] = \frac {\exp \left(o _ {i} \left[ x _ {i + 1} \right]\right)}{\sum_ {a = 1} ^ {\text {v o c a b - s i z e}} \exp \left(o _ {i} [ a ]\right)}. \tag {17}
$$

交叉熵损失通常是相对于 logits 向量 $o_i \in \mathbb{R}^{\mathrm{vocab\_size}}$ 和目标 $x_{i+1}$ 来定义的。<sup>7</sup>

实现交叉熵损失需要像 softmax 一样，在数值问题上小心处理。

# 问题 (cross_entropy): 实现交叉熵

交付物：编写一个函数来计算交叉熵损失，该函数接收预测的 logits $(o_i)$ 和目标 $(x_{i + 1})$，并计算交叉熵 $\ell_{i} = -\log \mathrm{softmax}(o_{i})[x_{i + 1}]$。你的函数应该处理以下问题：

- 减去最大元素以获得数值稳定性。
- 在可能的情况下抵消 log 和 exp。  
- 处理任何额外的批次维度，并在批次上返回平均值。与第 3.3 节一样，我们假设批次类维度始终排在词汇表大小维度之前。

实现 [adapters.run CROSS_entropy]，然后运行 uv run pytest -k test CROSS_entropy 来测试你的实现。

困惑度交叉熵足以用于训练，但当我们评估模型时，我们也想报告困惑度。对于长度为 $m$ 的序列，我们遭受交叉熵损失 $\ell_1,\ldots ,\ell_m$：

$$
\text {困惑度} = \exp \left(\frac {1}{m} \sum_ {i = 1} ^ {m} \ell_ {i}\right). \tag {18}
$$

# 4.2 SGD 优化器

现在我们有了损失函数，我们将开始探索优化器。最简单的基于梯度的优化器是随机梯度下降 (SGD)。我们从随机初始化的参数 $\theta_0$ 开始。然后对于每一步 $t = 0,\dots ,T - 1$，我们执行以下更新：

$$

\( \theta_{t + 1} \leftarrow \theta_{t} - \alpha_{t} \nabla L \left(\theta_{t}; B_{t}\right) \), (19)

其中 \(B_{t}\) 是从数据集 \(D\) 中采样的一个随机数据批次，学习率 \(\alpha_{t}\) 和批次大小 \(|B_{t}|\) 是超参数。

# 4.2.1 在 PyTorch 中实现 SGD

为了实现我们的优化器，我们将继承 PyTorch 的 torch.optim.Optimizer 类。Optimizer 子类必须实现两个方法：

<output>
 `__init__` 方法的 `params` 参数应该初始化你的优化器。这里，`params` 将是待优化的参数集合（或者参数组，如果用户希望为模型的不同部分使用不同的超参数，例如学习率）。确保将 `params` 传递给基类的 `__init__` 方法，它将存储这些参数以供 `step` 方法使用。你可以根据优化器接受额外的参数（例如，学习率是一个常见的参数），并将它们作为字典传递给基类构造函数，其中键是你为这些参数选择的名称（字符串）。
</output>

def step(self) 应该对参数进行一次更新。在训练循环中，这将在反向传播后调用，因此您可以访问最后一个批次的梯度。此方法应遍历每个参数张量 $\mathfrak{p}$ 并就地修改它们，即设置 p.data，它根据梯度 p.grad（如果存在）保存与该参数关联的张量，该张量表示损失相对于该参数的梯度。

PyTorch 优化器 API 有一些细微之处，因此最好通过示例来解释。为了使我们的示例更丰富，我们将实现 SGD 的一个细微变体，其中学习率在训练过程中衰减，从初始学习率 $\alpha$ 开始，并随着时间的推移逐渐减小步长：

$$
\theta_ {t + 1} = \theta_ {t} - \frac {\alpha}{\sqrt {t + 1}} \nabla L \left(\theta_ {t}; B _ {t}\right) \tag {20}
$$

让我们看看这个版本的 SGD 如何作为 PyTorch 优化器实现：

```python
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate:{lr}")
        defaults = {"lr": lr}
        super(SGD, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]  # 获取学习率

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # 获取与 p 关联的状态
                # 从状态中获取迭代次数，或使用初始值
                t = state.get("t", 0)
                grad = p.grad.data  # 获取损失相对于 p 的梯度
                # 更新权重张量
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1  # 增加迭代次数

        return loss
```

```javascript
for p in group["params"]:
    if p.grad is None:
        continue
    state = self.state[p]  # Get state associated with p
    t = state.get("t", 0)  # Get iteration number from the state, or initial value.
    grad = p.grad.data  # Get the gradient of loss with respect to p.
    p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
    state["t"] = t + 1  # Increment iteration number.

return loss
```

在 __init__ 中，我们将参数和默认超参数传递给基类构造函数（参数可能成组，每组有不同的超参数）。如果参数只是一个 torch.nn_PARAMETER 对象的集合，基类构造函数将创建一个组并为其分配默认超参数。然后，在 step 中，我们遍历每个参数组，然后遍历该组中的每个参数，并应用公式 20。在这里，我们将迭代次数作为与每个参数关联的状态：我们首先读取此值，在梯度更新中使用它，然后更新它。API 指定用户可以传入一个可调用闭包，在优化器步之前重新计算损失。我们使用的优化器不需要这个，但为了符合 API，我们添加了它。

为了看到这个工作过程，我们可以使用以下训练循环的最小示例：

```javascript
weights  $=$  torch.nn.Parameters(5 \* torch.random((10，10))) opt  $=$  SGD([weights],lr=1)
```

```python
for t in range(100):
    opt.zero_grad() # 重置所有可学习参数的梯度。
    loss = (weights**2).mean() # 计算标量损失值。
    print(loss.cpu().item())
    loss.backup() # 执行反向传播，计算梯度。
    opt.step() # 执行优化器步。
```

这是典型的训练循环结构：在每次迭代中，我们将计算损失并执行一次优化器步。在训练语言模型时，我们的可学习参数将来自模型（在PyTorch中，m.params()为我们提供了这个集合）。损失将在采样的数据批次上计算，但训练循环的基本结构将保持不变。

# 问题 (学习率调整): 调整学习率 (1 分)


正如我们将看到的，对训练影响最大的超参数之一是学习率。让我们在我们的玩具示例中实际看看。使用学习率的三个其他值：1e1、1e2 和 1e3，运行上面的 SGD 示例，仅进行 10 次训练迭代。这些学习率的损失会发生什么？它衰减得更快、更慢，还是发散（即在训练过程中增加）？

交付物：一两句话的响应，描述您观察到的行为。

# 4.3 AdamW

现代语言模型通常使用更复杂的优化器进行训练，而不是 SGD。最近使用的大多数优化器都是 Adam 优化器 [Kingma and Ba, 2015] 的派生。我们将使用 AdamW [Loshchilov and Hutter, 2019]，它在最近的工作中被广泛使用。AdamW 提出了一种对 Adam 的修改，通过添加权重衰减（在每次迭代中，我们将参数拉向 0）来改进正则化，

AdamW 是一种优化器，它将权重衰减与梯度更新解耦。我们将按照 Loshchilov 和 Hutter [2019] 的算法 2 中的描述来实现 AdamW。

AdamW 是有状态的：对于每个参数，它都会跟踪其一阶矩和二阶矩的运行估计。因此，AdamW 使用额外的内存来换取改进的稳定性和收敛性。除了学习率 $\alpha$ 之外，AdamW 还拥有一对超参数 $(\beta_{1},\beta_{2})$，用于控制矩估计的更新，以及一个权重衰减率 $\lambda$。典型应用将 $(\beta_{1},\beta_{2})$ 设置为 (0.9, 0.999)，但像 LLaMA [Touvron et al., 2023] 和 GPT-3 [Brown et al., 2020] 这样的大型语言模型通常使用 (0.9, 0.95) 进行训练。该算法可以写成如下形式，其中 $\epsilon$ 是一个很小的值（例如，$10^{-8}$），用于在 $v$ 中出现极小值时提高数值稳定性：

算法 1 AdamW 优化器
问题 (adamw)：实现 AdamW（2 分）  
问题 (adamwAccounting): 使用 AdamW 进行训练的资源核算（2 分）
```txt
init $(\theta)$ （初始化可学习参数）
$m\gets 0$ （一阶矩向量的初始值；形状与 $\theta$ 相同 $v\gets 0$ （二阶矩向量的初始值；形状与 $\theta$ 相同
for $t = 1,\ldots ,T$ do 采样数据批次 $B_{t}$ $g\gets \nabla_{\theta}\ell (\theta ;B_t)$ （计算当前时间步的损失梯度） $m\gets \beta_1m + (1 - \beta_1)g$ （更新一阶矩估计） $v\gets \beta_2v + (1 - \beta_2)g^2$ （更新二阶矩估计） $\alpha_{t}\gets \alpha \frac{\sqrt{1 - (\beta_{2})^{t}}}{1 - (\beta_{1})^{t}}$ （计算迭代 $t$ 的调整后 $\alpha$） 1- $(\beta_{1})^{t}$ $\theta \leftarrow \theta -\alpha_{t}\frac{m}{\sqrt{v} + \epsilon}$ （更新参数） $\theta \gets \theta -\alpha \lambda \theta$ （应用权重衰减）
end for
```

请注意，$t$ 从 1 开始。你现在将实现这个优化器。

交付物：实现 AdamW 优化器，作为 torch.optim.Optimizer 的子类。你的类应该在 __init__ 中接收学习率 $\alpha$，以及 $\beta$、$\epsilon$ 和 $\lambda$ 超参数。为了帮助你保持状态，基类 Optimizer 提供了一个字典 self.state，它将 nn.Parameters 对象映射到一个字典，该字典存储了你需要的关于该参数的任何信息（对于 AdamW，这将是动量估计）。实现 [adapters.get_adamw_cls]，并确保它通过 uv_run.pytest -k test_adamw。

让我们计算一下运行 AdamW 所需的内存和计算量。假设我们对每个张量都使用 float32。

(a) 运行 AdamW 需要多少峰值内存？根据参数、激活值、梯度和优化器状态的内存使用情况分解您的答案。用 batch_size 和模型超参数（vocab_size、context_length、num_layers、d_model、num_heads）表示您的答案。假设 $\text{d}_\text{ff} = 4 \times \text{d}_\text{model}$。

为简单起见，在计算激活值的内存使用时，仅考虑以下组件：

- Transformer 块

RMSNorm(s)

- 多头自注意力子层： $QKV$ 投影， $Q^{\top}K$ 矩阵乘法，softmax，值的加权和，输出投影。
- 按位置的馈forward： $W_{1}$ 矩阵乘法，SiLU， $W_{2}$ 矩阵乘法

- 最终 RMSNorm
输出嵌入
- logits 上的交叉熵

交付成果：参数、激活值、梯度和优化器状态以及总计的代数表达式。

(b) 为 GPT-2 XL 模型实例化您的答案，得到一个仅取决于 batch_size 的表达式。您能使用的最大 batch size 是多少，才能仍然适合 80GB 内存？

交付物：一个形如 $a \cdot \text{batch\_size} + b$ 的表达式，其中 $a, b$ 为数值，以及一个代表最大 batch size 的数字。

(c) 运行一步 AdamW 需要多少 FLOPs？

交付物：一个代数表达式，附带简要说明。

(d) 模型 FLOPs 利用率 (MFU) 定义为观察到的吞吐量（每秒词元数）相对于硬件理论峰值 FLOPs 吞吐量的比率 [Chowdhery et al., 2022]。一块英伟达 A100 GPU 的 float32 操作理论峰值为 19.5 teraFLOP/s。假设你能获得 50% 的 MFU，在单块 A100 上训练 GPT-2 XL 400K 步，批次大小为 1024，需要多长时间？根据 Kaplan et al. [2020] 和 Hoffmann et al. [2022] 的假设，反向传播的 FLOPs 是前向传播的两倍。

交付成果：训练所需天数，并附简要理由。

# 4.4 学习率调度

在训练过程中，导致损失下降最快的学习率值通常会发生变化。在训练 Transformer 时，通常会使用学习率调度，即我们从一个较大的学习率开始，在开始时进行更快的更新，并随着模型的训练逐渐将其衰减到一个较小的值<sup>8</sup>。在此次作业中，我们将实现用于训练 LLaMA [Touvron et al., 2023] 的余弦退火调度。

调度器只是一个函数，它接收当前步数 $t$ 和其他相关参数（例如初始和最终学习率），并返回在步数 $t$ 的梯度更新中使用的学习率。最简单的调度器是常数函数，它对于任何 $t$ 都会返回相同的学习率。

余弦退火学习率调度器需要 (i) 当前迭代次数 $t$、(ii) 最大学习率 $\alpha_{\mathrm{max}}$、(iii) 最小（最终）学习率 $\alpha_{\mathrm{min}}$、(iv) 热身迭代次数 $T_w$ 和 (v) 余弦退火迭代次数 $T_c$。迭代次数 $t$ 时的学习率定义为：

(热身) 如果 $t < T_w$，则 $\alpha_t = \frac{t}{T_w} \alpha_{\max}$。

(余弦退火) 如果 $T_w \leq t \leq T_c$，则 $\alpha_t = \alpha_{\min} + \frac{1}{2}\left(1 + \cos\left(\frac{t - T_w}{T_c - T_w}\pi\right)\right)\left(\alpha_{\max} - \alpha_{\min}\right)$。

(退火后) 如果 $t > T_{c}$，则 $\alpha_{t} = \alpha_{\min}$。

# 问题 (learning_rate_schedule): 实现带热身的余弦学习率调度器


编写一个函数，该函数接受 $t$、$\alpha_{\mathrm{max}}$、$\alpha_{\mathrm{min}}$、$T_w$ 和 $T_c$，并根据上述调度器返回学习率 $\alpha_t$。然后实现 [adapters.get_lr_cosine_schedule]，并确保它通过 uv run pytest -k test_get_lr_cosine_schedule。

# 4.5 梯度裁剪

在训练过程中，我们有时会遇到产生大梯度的训练样本，这会使训练不稳定。为了缓解这种情况，实践中常用的技术是梯度裁剪。其思想是在每次反向传播后，在执行优化器步之前，对梯度的范数施加一个限制。

给定（所有参数的）梯度 $g$，我们计算其 $\ell_2$ 范数 $\| g\| _2$。如果该范数小于最大值 $M$，则保持 $g$ 不变；否则，将 $g$ 按因子 $\frac{M}{\|g\|_2 + \epsilon}$ 进行缩放（其中添加了一个小的 $\epsilon$，例如 $10^{-6}$，以提高数值稳定性）。请注意，结果范数将略小于 $M$。

# 问题 (gradient_clipping): 实现梯度裁剪 (1 分)

编写一个实现梯度裁剪的函数。您的函数应接收一个参数列表和一个最大 $\ell_2$ 范数。它应该就地修改每个参数梯度。使用 $\epsilon = 10^{-6}$（PyTorch 默认值）。然后，实现适配器 [adapters.run_gradment_clipping]，并确保它通过 uv run pytest -k test_gradment_clipping。

# 5 训练循环

现在我们将把我们迄今为止构建的主要组件放在一起：分词数据、模型和优化器。

# 5.1 数据加载器

token=>词元
tokenizer=>分词器
<|endoftext|>=><|endoftext|>

分词后的数据（例如，您在 tokenizer_experiments 中准备的数据）是单个词元序列 $x = (x_{1},\ldots ,x_{n})$。尽管源数据可能由单独的文档组成（例如，不同的网页或源代码文件），但一种常见的做法是将所有这些文档连接成一个单一的词元序列，并在它们之间添加一个分隔符（例如 <|endoftext|> 词元）。

数据加载器将此转换为一批一批的流，其中每批包含 $m$ 长度的 $B$ 个序列，并配有相应的下一个词元，长度也为 $m$。例如，对于 $B = 1, m = 3$，$( [x_2, x_3, x_4], [x_3, x_4, x_5] )$ 将是一个可能的批次。

以这种方式加载数据有几个原因可以简化训练。首先，任何 $1 \leq i < n - m$ 都可以得到一个有效的训练序列，因此采样序列是微不足道的。由于所有训练序列的长度都相同，因此无需填充输入序列，这可以提高硬件利用率（也通过增加批次大小 $B$ 来实现）。最后，我们也不需要完全加载整个数据集来采样训练数据，这使得处理可能无法放入内存的大型数据集变得容易。

# 问题（数据加载）：实现数据加载（2分）

Deliverable：编写一个函数，该函数接受一个 NumPy 数组 $x$（整数数组，包含词元 ID）、batch_size、context_length 和一个 PyTorch 设备字符串（例如 'cpu' 或 'CUDA:0'），并返回一个张量对：采样的输入序列和相应的下一个词元目标。两个张量都应具有形状 (batch_size, context_length)，包含词元 ID，并且都应放置在请求的设备上。要使用我们提供的测试来测试您的实现，您首先需要实现 [adapters.run_get_batch] 中的测试适配器。然后，运行 uv run pytest -k test_get_batch 来测试您的实现。

# 低资源/降级提示：在 CPU 或 Apple Silicon 上加载数据

如果您计划在 CPU 或 Apple Silicon 上训练您的 LM，您需要将数据移动到正确的设备（同样，稍后您应该为您的模型使用相同的设备）。

如果您使用的是 CPU，可以使用 'cpu' 设备字符串；如果您使用的是 Apple Silicon（M*芯片），可以使用 'mps' 设备字符串。

有关 MPS 的更多信息，请参阅以下资源：

- https://developer.apple.com/metals/pytorch/
- https://pytorch.org/docs/main/notes/mps.html

如果数据集太大而无法加载到内存中怎么办？我们可以使用一个名为 mmap 的 Unix 系统调用，它将磁盘上的文件映射到虚拟内存，并在访问该内存位置时惰性地加载文件内容。因此，您可以“假装”整个数据集都在内存中。Numpy 通过 np.memmap（或者在 np.load 中使用标志 mmap_mode='r'，如果您最初使用 np.save 保存了数组）来实现这一点，它将返回一个类似 numpy 数组的对象，该对象会在您访问它们时按需加载条目。在训练期间从数据集中采样（即 numpy 数组）时，请确保以内存映射模式加载数据集（通过 np.memmap 或在 np.load 中使用标志 mmap_mode='r'，具体取决于您保存数组的方式）。确保您还指定一个与您正在加载的数组匹配的 dtype。显式验证内存映射数据是否正确可能很有帮助（例如，不包含超出预期词汇量大小的值）。

# 5.2 检查点

除了加载数据，我们还需要在训练过程中保存模型。在运行作业时，我们通常希望能够恢复因某种原因中途停止的训练运行（例如，由于作业超时、机器故障等）。即使一切顺利，我们也可能希望稍后能够访问中间模型（例如，事后研究训练动态、从不同训练阶段的模型中采样等）。

一个检查点应该包含我们恢复训练所需的所有状态。我们当然希望至少能够恢复模型的权重。如果使用有状态的优化器（例如 AdamW），我们还需要保存优化器的状态（例如，对于 AdamW，是动量估计）。最后，要恢复学习率调度器，我们需要知道停止时的迭代次数。PyTorch 可以轻松保存所有这些：每个 nnModule 都有一个 state_dict() 方法，该方法返回一个包含所有可学习权重的字典；我们可以稍后使用其对应的 load_state_dict() 方法恢复这些权重。对于任何 nn_optimOptimizer 也是如此。最后，torch.save(obj, dest) 可以将一个对象（例如，一个包含张量值或普通 Python 对象（如整数）的字典）转储到一个文件（路径）或类文件对象，然后可以使用 torch.load(src) 将其加载回内存。

# 问题（检查点）：实现模型检查点（1 分）

实现以下两个函数来加载和保存检查点：

def save_checkpoint(model, optimizer, iteration, out) 应将前三个参数的所有状态转储到类文件对象 out 中。您可以使用模型和优化器各自的 state_dict 方法来获取它们的相关状态，并使用 torch.save(obj, out) 将 obj 转储到 out 中（PyTorch 支持路径或类文件对象）。通常选择 obj 为字典，但只要您以后能够加载检查点，您就可以使用任何格式。

此函数需要以下参数：

model: torch(nnModule

optimizer: torch.optim.Optimizer

iteration: int

out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]

def load_checkpoint(src, model, optimizer) 应从 src（路径或类文件对象）加载检查点，然后从该检查点恢复模型和优化器状态。你的函数应返回保存到检查点的迭代次数。你可以使用 torch.load(src) 来恢复你在 save_checkpoint 实现中保存的内容，并使用 model 和 optimizers 中的 load_state_dict 方法将它们恢复到之前的状态。

此函数期望以下参数：

src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]

model: torch(nnModule

optimizer: torch.optim.Optimizer

实现 [adapters.run_save_checkpoint] 和 [adapters.run_load_checkpoint] 适配器，并确保它们通过 uv run pytest -k test_checkpointing。

# 5.3 训练循环

现在，是时候将你实现的所有组件整合到主训练脚本中了。让不同超参数的训练运行（例如，通过将它们作为命令行参数传入）更容易启动将会带来回报，因为之后你将多次进行这些操作来研究不同选择如何影响训练。

# 问题 (training_together)：整合 (4 分)

交付成果：编写一个脚本，该脚本运行一个训练循环，在用户提供的输入上训练你的模型。特别是，我们建议你的训练脚本至少允许：

- 配置和控制各种模型及优化器超参数的能力。
- 使用 np.memmap 内存高效地加载训练和验证大型数据集。
- 将检查点序列化到用户提供的路径。
- 定期记录训练和验证性能（例如，记录到控制台和/或外部服务，如 Weights and Biases）。<sup>a</sup>

# 6 生成文本

现在我们可以训练模型了，我们需要的最后一块就是从模型生成文本的能力。回想一下，语言模型接收一个长度为 (sequence_length) 的（可能分批的）整数序列，并生成一个大小为 (sequence_length × vocab_size) 的矩阵，其中序列的每个元素都是一个概率分布，预测该位置之后的下一个词。我们现在将编写几个函数来实现这种新序列的采样方案。

Softmax 按照标准约定，语言模型的输出是最后一个线性层（“logits”）的输出，因此我们必须通过 softmax 操作将其转换为归一化概率，我们在公式10中已经见过。

解码

为了从我们的模型中生成文本（解码），我们将向模型提供一个前缀词元序列（“提示”），并要求它生成一个词汇表上的概率分布，以预测序列中的下一个词。然后，我们将从词汇表项上的这个分布中采样，以确定下一个输出词元。

具体来说，解码过程的一个步骤应该接收一个序列 $x_{1\dots t}$，并通过以下方程返回一个词元 $x_{t + 1}$：

$$
\begin{array}{l} P (x _ {t + 1} = i \mid x _ {1 \dots t}) = \frac {\exp (v _ {i})}{\sum_ {j} \exp (v _ {j})} \\ v = \operatorname {T r a n s f o r m e r L M} (x _ {1 \dots t}) _ {t} \in \mathbb {R} ^ {\text {v o c a b - s i z e}} \\ \end{array}
$$

其中 TransformerLM 是我们的模型，它接收一个 sequence_length 的序列作为输入，并产生一个 (sequence_length × vocab_size) 大小的矩阵，我们取这个矩阵的最后一个元素，因为我们正在寻找第 $t$ 个位置的下一个词预测。

通过重复地从这些单步条件中采样（将我们先前生成的输出词元附加到下一个解码时间步的输入中），直到我们生成序列结束词元 $< \mid$ endoftext $\mid>$ （或用户指定的要生成的词元的最大数量），我们可以得到一个基本的解码器。

解码器技巧 我们将尝试使用小型模型，而小型模型有时会生成质量非常低的文本。两个简单的解码器技巧可以帮助解决这些问题。首先，在温度缩放中，我们用温度参数 $\tau$ 修改我们的 softmax，其中新的 softmax 是

$$
$\operatorname{softmax}(v, \tau)_{i}=\frac{\exp \left(v_{i} / \tau\right)}{\sum_{j=1}^{|\text {vocab\_size}|} \exp \left(v_{j} / \tau\right)} \tag{24}$

注意，将 $\tau \to 0$ 设置为使 $v$ 的最大元素占主导地位，并且 softmax 的输出成为集中在该最大元素上的独热向量。

其次，另一个技巧是核采样或 top- $p$ 采样，我们通过截断低概率词来修改采样分布。令 $q$ 为我们从大小为 (vocab_size) 的（温度缩放的）softmax 得到的概率分布。具有超参数 $p$ 的核采样根据以下方程产生下一个词元：

$$
P(x_{t+1}=i | q) = \left\{ \begin{array}{ll} \frac{q_i}{\sum_{j \in V(p)} q_j} & \text {if } i \in V(p) \\ 0 & \text {otherwise} \end{array} \right.
$$

其中 $V(p)$ 是最小的索引集，使得 $\sum_{j \in V(p)} q_j \geq p$。你可以通过首先按大小对概率分布 $q$ 进行排序，然后选择最大的词汇元素直到达到目标水平 $\alpha$ 来轻松计算此数量。

# 问题（解码）：解码（3分）

交付成果：实现一个从语言模型解码的函数。我们建议您支持以下功能：

- 为用户提供的提示生成补全（即，输入一些 $x_{1\dots t}$ 并采样补全，直到遇到一个 $< | \text{endoftext} | >$ 词元）。
- 允许用户控制生成的词元的最大数量。
- 给定所需的温度值，在采样之前将预测的下一个词分布应用 softmax 温度缩放。
- Top-p 采样（Holtzman 等人，2020；也称为 nucleus 采样），给定用户指定的阈值。

# 7 实验

现在是时候将所有内容整合起来，并在预训练数据集上训练（小型）语言模型了。

# 7.1 如何运行实验和交付成果

理解 Transformer 架构组件背后原理的最佳方法是亲自修改和运行它。没有替代动手实践的经验。

为此，能够快速、一致地进行实验并记录所做的事情非常重要。为了快速进行实验，我们将在小型模型（1700 万参数）和简单数据集（TinyStories）上运行许多实验。为了保持一致性，您将以系统化的方式剥离组件并改变超参数，为了记录，我们将要求您提交实验日志以及与每个实验相关的学习曲线。

为了能够提交损失曲线，请确保定期评估验证损失，并记录步数和挂钟时间。您可能会发现 Weights and Biases 等日志记录基础设施很有帮助。

# 问题 (experiment_log): 实验日志 (3 分)

对于您的训练和评估代码，请创建实验跟踪基础设施，使您能够跟踪实验和损失曲线相对于梯度步数和挂钟时间。

交付物：实验日志记录基础设施代码以及本节下面分配问题的实验日志（您尝试过的所有内容的文档）。

# 7.2 TinyStories

我们将从一个非常简单的数据集（TinyStories；Eldan and Li, 2023）开始，模型将在此数据集上快速训练，并且我们可以看到一些有趣的行为。获取此数据集的说明在第 1 节。下面是该数据集外观的示例。

<output>
# 示例 (tinystories_example): TinyStories 的一个示例

从前有个小男孩叫蒂姆。蒂姆喜欢探索周围的世界。他看到了许多令人惊叹的东西，比如商店里陈列的美丽花瓶。有一天，蒂姆在商店里散步时，发现了一个非常特别的花瓶。当蒂姆看到它时，他惊呆了！他说：“哇，这真是一个令人惊叹的花瓶！我能买下它吗？”店主笑了笑说：“当然可以。你可以把它带回家，向所有朋友展示它有多么令人惊叹！”于是蒂姆把花瓶带回了家，他为此感到非常自豪！他叫来朋友们，向他们展示了这个令人惊叹的花瓶。他的朋友们都觉得这个花瓶很漂亮，简直不敢相信蒂姆有多幸运。这就是蒂姆在商店里找到一个令人惊叹的花瓶的故事！

超参数调整 我们将为您提供一些非常基础的超参数作为起点，并要求您找到一些适用于其他参数的有效设置。
</output>

vocab_size 10000。典型的词汇表大小在十万到数十万之间。你应该改变这个值，看看词汇表和模型行为如何变化。

context_length 256。像TinyStories这样简单的数据集可能不需要很长的序列长度，但对于后面的OpenWebText数据，你可能需要改变它。尝试改变它，看看对每次迭代的运行时间和最终困惑度的影响。

d_model 512。这比许多小型Transformer论文中使用的768个维度要小一些，但这会使事情更快。

d_ff 1344。这大约是 $\frac{8}{3}$ d_model，同时是64的倍数，这有利于GPU性能。

RoPE theta参数 $\Theta$ 10000。

层数和头数 4层，16头。总的来说，这将提供大约17M个非嵌入参数，这是一个相当小的Transformer。

处理的总词元数 327,000,000（你的批次大小 $\times$ 总步数 $\times$ 上下文长度应大致等于此值）。

你需要进行一些试错来找到以下其他超参数的良好默认值：学习率、学习率预热、其他 AdamW 超参数 $(\beta_{1},\beta_{2},\epsilon)$ 和权重衰减。你可以在 Kingma 和 Ba [2015] 中找到这些超参数的一些典型选择。

整合起来 现在你可以将所有内容整合起来，获取一个训练好的 BPE 分词器，对训练数据集进行分词，并在你编写的训练循环中运行它。重要提示：如果你的实现是正确且高效的，上述超参数应该能在 1 块 H100 GPU 上实现大约 30-40 分钟的运行时间。如果你的运行时间长得多，请检查并确保你的数据加载、检查点或验证损失代码没有成为你运行时间的瓶颈，并且你的实现已正确分批。

调试模型架构的技巧和窍门 我们强烈建议您熟悉 IDE 内置的调试器（例如 VSCode/PyCharm），与使用 print 语句进行调试相比，这可以节省您的时间。如果您使用文本编辑器，可以使用类似 pdb 的工具。调试模型架构时，还有一些其他好的做法：

- 开发任何神经网络架构时，一个常见的首要步骤是使其过拟合到单个小批量。如果您的实现正确，您应该能够快速将训练损失降至接近零。
- 在模型的各个组件中设置调试断点，并检查中间张量的形状，以确保它们符合您的预期。
- 监控激活、模型权重和梯度的范数，以确保它们不会爆炸或消失。

# 问题（learning_rate）：调整学习率（3 分）（4 H100 小时）

学习率是最重要的超参数之一。以你训练的基础模型为例，回答以下问题：

(a) 对学习率进行超参数扫描，并报告最终损失（如果优化器发散，请注明发散情况）。

交付物：与多个学习率相关的学习曲线。解释你的超参数搜索策略。

交付物：一个在 TinyStories 上验证损失（每词元）不超过 1.45 的模型。

# 低资源/降级技巧：在 CPU 或 Apple Silicon 上训练较少步数

如果你在 CPU 或 MPS 上运行，你应该将处理的总词元数减少到 40,000,000，这足以生成相当流畅的文本。你也可以将目标验证损失从 1.45 提高到 2.00。

使用经过优化的学习率，在 M3 Max 芯片和 36 GB RAM 上运行我们的解决方案代码，我们使用的批次大小 $\times$ 总步数 $\times$ 上下文长度 $= 32 \times 5000 \times 256 = 40,960,000$ 个词元，在 CPU 上耗时 1 小时 22 分钟，在 MPS 上耗时 36 分钟。在第 5000 步时，我们达到了 1.80 的验证损失。

一些额外的建议：

- 当使用 $X$ 训练步数时，我们建议调整余弦学习率衰减计划，使其在第 $X$ 步精确地终止衰减（即达到最小学习率）。
- 使用 MPS 时，不要使用 TF32 核心，即不要设置 `torch.set_float32_matmul_precision('high')`，这与使用 CUDA 设备时不同。我们尝试在 MPS（torch 版本 2.6.0）上启用 TF32 核心，发现后端会默默地使用损坏的核心，导致训练不稳定。

- 您可以通过使用 `torch.compile` 对模型进行 JIT 编译来加速训练。具体来说：

- 在 CPU 上，使用以下命令编译您的模型

$$

模型 = torch.compile(model)

- 在 mps 上，您可以通过以下方式在一定程度上优化反向传播：

模型 = torch.compile(model, backend="aot_eager")

截至 torch 版本 2.6.0，在 mps 上不支持使用 Inductor 进行编译。

(b) 普遍的看法是，最佳学习率是“处于稳定性边缘”。研究学习率发散点与您的最佳学习率之间的关系。

交付成果：学习曲线，其中包含递增的学习率，至少包含一次发散运行，并分析其与收敛率的关系。

现在，让我们改变批次大小，看看训练会发生什么。批次大小很重要——它们通过执行更大的矩阵乘法使我们的 GPU 获得更高的效率，但我们是否总是希望批次大小很大呢？让我们进行一些实验来找出答案。

# 问题（batch_size 实验）：Batch size 变化（1 个点）（2 个 H100 小时）

将 batch size 从 1 调整到 GPU 内存限制。尝试几个中间的 batch size，包括 64 和 128 等典型大小。

交付物：不同 batch size 运行的学习曲线。如有必要，应重新优化学习率。

交付物：几句话讨论您关于 batch size 及其对训练影响的发现。

有了 decoder，我们现在可以生成文本了！我们将从模型中生成并查看其效果。作为参考，您的输出应该至少和下面的示例一样好。

# 示例（ts_generate_example）：TinyStories 语言模型的示例输出

Once upon a time, there was a pretty girl named Lily. She loved to eat gum, especially the big black one. One day, Lily's mom asked her to help cook dinner. Lily was so excited! She loved to help her mom. Lily's mom made a big pot of soup for dinner. Lily was so happy and said, "Thank you, Mommy! I love you." She helped her mom pour the soup into a big bowl. After dinner, Lily's mom made some yummy soup. Lily loved it! She said, "Thank you, Mommy! This soup is so yummy!" Her mom smiled and said, "I'm glad you like it, Lily." They finished cooking and continued to cook together. The end.

# Low-Resource/Downscaling Tip: Generate text on CPU or Apple Silicon

If instead you used the low-resource configuration with 40M tokens processed, you should see generations that still resemble English but are not as fluent as above. For example, our sample output from a TinyStories language model trained on 40M tokens is below:

从前，有一个叫苏的小女孩。苏有一颗她非常喜欢的牙齿。这是他最好的头。有一天，苏出去散步，遇到了一只瓢虫！他们成了好朋友，一起在小路上玩耍。

“嘿，波莉！我们出去吧！”蒂姆说。苏看着天空，发现很难找到跳舞闪耀的方法。她笑了，同意帮助说话！

当苏看着天空移动时，它是什么。她

这是精确的问题陈述以及我们的要求：

# 问题（生成）：生成文本（1 分）

使用您的解码器和训练好的检查点，报告您的模型生成的文本。您可能需要调整解码器参数（温度、top-p 等）以获得流畅的输出。

可交付成果：文本转储，至少包含 256 个词元（或直到第一个 $<$ |endoftext|> 词元），以及关于此输出流畅度的简短评论，以及至少两个影响此输出好坏的因素。

# 7.3 消融和架构修改

理解 Transformer 的最佳方法是实际修改它并观察其行为。我们现在将进行一些简单的消融和修改。

消融 1：层归一化 有时人们说层归一化对于 Transformer 训练的稳定性很重要。但也许我们想冒险一搏。让我们移除我们每个 Transformer 块中的 RMSNorm，看看会发生什么。

# 问题 (layer_norm_ablation)：移除 RMSNorm 并进行训练（1 分）（1 H100 小时）

移除你 Transformer 中的所有 RMSNorm 并进行训练。在之前的最优学习率下会发生什么？你能通过使用较低的学习率来获得稳定性吗？

交付物：移除 RMSNorm 并进行训练时的学习曲线，以及最佳学习率的学习曲线。

交付物：几句话评论 RMSNorm 的影响。

现在让我们研究另一个乍一看似乎是任意选择的层归一化。Pre-norm Transformer 块定义为

$$
z = x + \text {M u l t i H e a d e d S e l f A t t e n t i o n} (\operatorname {R M S N o r m} (x))

$$
y = z + \operatorname {F F N} (\operatorname {R M S N o r m} (z)).
$$

这是对原始 Transformer 架构的少数“共识”修改之一，它使用了后归一化方法，如

$$
z = \operatorname {R M S N o r m} (x + \text {M u l t i H e a d e d S e l f A t t e n t i o n} (x))
$$

$$
y = \operatorname {R M S N o r m} (z + \operatorname {F F N} (z)).
$$

让我们回到后归一化方法，看看会发生什么。

# 问题 (pre_norm_ablation): 实现后归一化并进行训练 (1 分) (1 H100 小时)

将您的预归一化 Transformer 实现修改为后归一化。使用后归一化模型进行训练，看看会发生什么。

交付成果：后归一化 Transformer 的学习曲线，与预归一化 Transformer 进行比较。

我们看到层归一化对 Transformer 的行为有重大影响，甚至层归一化的位置也很重要。

消融实验 2：位置嵌入 我们接下来将研究位置嵌入对模型性能的影响。具体来说，我们将比较我们的基础模型（带有 RoPE）与完全不包含位置嵌入（NoPE）的模型。事实证明，仅解码器的 Transformer，即那些具有我们已实现的因果掩码的 Transformer，理论上可以在不显式提供位置嵌入的情况下推断相对或绝对位置信息 [Tsai et al., 2019, Kazemnejad et al., 2023]。我们现在将通过实验测试 NoPE 与 RoPE 相比的性能。

# 问题 (no_pos_emb)：实现 NoPE（1 分）（1 H100 小时）

修改您带有 RoPE 的 Transformer 实现，完全移除位置嵌入信息，看看会发生什么。

交付物：一个学习曲线，比较 RoPE 和 NoPE 的性能。

消融实验 3：SwiGLU vs. SiLU
接下来，我们将遵循 Shazeer [2020] 的方法，通过比较 SwiGLU 前馈网络与使用 SiLU 激活但没有门控线性单元 (GLU) 的前馈网络的性能，来测试门控在前馈网络中的重要性：

$$
\mathrm {F F N} _ {\mathrm {S i L U}} (x) = W _ {2} \mathrm {S i L U} \left(W _ {1} x\right). \tag {25}
$$

回想一下，在我们 SwiGLU 的实现中，我们将内部前馈层的维度设置为大约 $d_{\mathrm{ff}} = \frac{8}{3} d_{\mathrm{model}}$ （同时确保 $d_{\mathrm{ff}} \mod 64 = 0$，以利用 GPU 张量核心）。在你的 $\mathrm{FFN}_{\mathrm{SiLU}}$ 实现中，你应该将 $d_{\mathrm{ff}} = 4 \times d_{\mathrm{model}}$，以大致匹配 SwiGLU 前馈网络的参数数量（它有三个权重矩阵而不是两个）。

# 问题 (swiglu_ablation)：SwiGLU vs. SiLU (1 分) (1 H100 小时)

<output>
交付成果：一个学习曲线，比较 SwiGLU 和 SiLU 前馈网络的性能，参数数量大致匹配。

# 低资源/降尺度技巧：GPU 资源有限的在线学生应在 TinyStories 上测试修改

在本次作业的剩余部分，我们将转向更大规模、更嘈杂的网络数据集（Open-WebText），尝试架构修改，并（可选地）提交到课程排行榜。

在 OpenWebText 上将一个 LM 训练到流利需要很长时间，因此我们建议 GPU 访问受限的在线学生继续在 TinyStories 上测试修改（使用验证损失作为评估性能的指标）。

# 7.4 在 OpenWebText 上运行

我们现在将转向一个由网络爬取创建的更标准的预训练数据集。Open-WebText [Gokaslan et al., 2019] 的一小部分也作为单个文本文件提供：有关如何访问此文件的信息，请参阅第 1 节。
</output>

Here is an example from OpenWebText. Note how the text is much more realistic, complex, and varied. You may want to look through the training dataset to get a sense of what training data looks like for a webscraped corpus.

# Example (owt_example): One example from OWT

棒球前景技术总监哈里·帕夫利迪斯在雇佣乔纳森·贾奇时冒了风险。

帕夫利迪斯知道，正如艾伦·施瓦茨在《数字游戏》中所写的那样，“美国文化中没有哪个角落比棒球运动员的表现更能被精确计算、更被热情量化。” 经过几次点击，你就可以发现诺亚·辛德加德的快球在飞向本垒的途中每分钟旋转超过 2100 次，纳尔逊·克鲁兹在 2016 年的合格击球手中拥有最高的比赛平均击球速度，以及无数其他似乎摘自电子游戏或科幻小说的小细节。不断增长的数据海洋赋予了棒球文化中一个日益重要的参与者力量：分析爱好者。

伴随这种赋权而来的是更多的审视——不仅是对测量方法，也是对它们背后的人员和出版物。通过《棒球前景》，帕夫利迪斯深知量化不完美所带来的负面影响。他也知道该网站的接球指标需要重新调整，并且需要一个有学识的人——一个能够解决复杂统计建模问题的人——来完成这项工作。

“他让我们大吃一惊。”哈里·帕夫利迪斯

帕夫利迪斯凭直觉认为，根据后者（Judge）的写作和他俩在一个网站赞助的球场活动中的互动，Judge“领会了”其中的要点。此后不久，两人在一次饮酒时进行了交谈。帕夫利迪斯的直觉得到了证实。Judge非常适合这个职位——更确切地说，他是一个乐意接受这个职位的人。“我问了很多人，”帕夫利迪斯说，“他是唯一一个敢于承担这项工作的人。” [...]

注意：您可能需要为本次实验重新调整超参数，例如学习率或批次大小。

# 问题 (mainExperiment): 在 OWT 上进行实验 (2 分) (3 H100 小时)

在 OpenWebText 上训练你的语言模型，模型架构和总训练迭代次数与 TinyStories 相同。这个模型表现如何？

交付物：你的语言模型在 OpenWebText 上的学习曲线。描述与 TinyStories 的损失差异——我们应该如何解释这些损失？

交付物：从 OpenWebText LM 生成的文本，格式与 TinyStories 的输出相同。这段文本的流畅度如何？为什么即使我们拥有与 TinyStories 相同的模型和计算预算，输出质量却更差？

# 7.5 你自己的修改 + 排行榜

恭喜你走到这一步。你快完成了！你现在将尝试改进 Transformer 架构，并看看你的超参数和架构与其他同学相比如何。

排行榜规则 除了以下几点外，没有其他限制：

<output>
 运行时 您的提交最多可以在 H100 上运行 1.5 小时。您可以通过在 slurm 提交脚本中设置 --time=01:30:00 来强制执行此操作。

数据 您只能使用我们提供的 OpenWebText 训练数据集。

否则，您可以随心所欲。

如果您正在寻找一些实现想法，可以查看以下资源：

- 最先进的开源 LM 系列，例如 Llama 3 [Grattafori et al., 2024] 或 Qwen 2.5 [Yang et al., 2024]。
</output>  
- NanoGPT 速度运行仓库 (https://github.com/KellerJordan/modded-nanogpt)，社区成员在此发布了许多用于“速度运行”小型语言模型预训练的有趣修改。例如，一个可以追溯到原始 Transformer 论文的常见修改是将输入和输出嵌入的权重绑定在一起（参见 Vaswani et al. [2017]（第 3.4 节）和 Chowdhery et al. [2022]（第 2 节））。如果您尝试权重绑定，可能需要减小嵌入/LM 头初始化的标准差。

您可能希望在 OpenWebText 的一小部分数据或 TinyStories 上测试这些修改，然后再尝试完整的 1.5 小时运行。

需要注意的是，我们确实注意到，您可能会发现在此排行榜中运行良好的某些修改可能无法推广到更大规模的预训练。我们将在课程的缩放定律单元中进一步探讨这个想法。

# 问题（排行榜）：排行榜（6 分）（10 H100 小时）

您将在上述排行榜规则下训练一个模型，目标是在 1.5 个 H100 小时内最小化您的语言模型的验证损失。

交付物：记录的最终验证损失，一个清晰显示时钟时间 x 轴（小于 1.5 小时）的相关学习曲线，以及对您所做工作的描述。我们期望排行榜提交的成绩至少要优于 5.0 损失的朴素基线。在此处提交到排行榜：https://github.com/stanford-cs336/assignment1-basics-leaderboard。

# References

Ronen Eldan and Yuanzhi Li. TinyStories: How small can language models be and still speak coherent English?, 2023. arXiv:2305.07759.
Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, and Stefanie Tellex. OpenWebText corpus. http://Skylion007.github.io/OpenWebTextCorpus, 2019.
Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. In Proc. of ACL, 2016.  
王昌汉、赵景贤和顾嘉涛。Neural machine translation with byte-level subwords, 2019. arXiv:1909.03341.
菲利普·盖奇。A new algorithm for data compression. C Users Journal, 12(2):23-38, February 1994. ISSN 0898-9788.
亚历克·拉德福德、吴杰、雷翁·柴尔德、David Luan、达里奥·阿莫代和伊利亚·苏茨克维。Language models are unsupervised multitask learners, 2019.
亚历克·拉德福德、卡尔蒂克·纳拉辛汉、蒂姆·萨利曼斯和伊利亚·苏茨克维。Improving language understanding by generative pre-training, 2018.
阿希什·瓦斯瓦尼、诺姆·沙泽尔、妮基·帕玛、雅各布·乌什科雷特、里昂·琼斯、艾丹·N·戈麦斯、卢卡斯·凯泽和伊利亚·波洛苏欣。Attention is all you need. In Proc. of NeurIPS, 2017.
阮Q. 阮和朱利安·萨拉查。Transformers without tears: Improving the normalization of self-attention. In Proc. of IWSWLT, 2019.  
熊瑞宾、杨云昌、何迪、郑凯、郑树新、邢晨、张辉帅、兰艳艳、王利伟和刘铁岩。On layer normalization in the Transformer architecture。In Proc. of ICML, 2020。
鲍继荣、杰米·瑞安·基罗斯和杰弗里·E·辛顿。Layer normalization, 2016。arXiv:1607.06450。
雨果·图夫龙、蒂博·拉夫里尔、戈蒂尔·伊扎卡德、泽维尔·马蒂内、玛丽-安妮·拉肖、蒂莫泰·拉科鲁瓦、巴蒂斯特·罗齐耶尔、纳曼·戈亚尔、埃里克·汉布罗、费萨尔·阿兹哈尔、奥雷利安·罗德里格斯、阿曼德·朱林、爱德华·格雷夫和纪尧姆·兰普尔。Llama: Open and efficient foundation language models, 2023。arXiv:2302.13971。
张彪和里科·森里奇。Root mean square layer normalization。In Proc. of NeurIPS, 2019。  
亚伦·格拉塔夫蒂奥里、阿比曼尤·杜贝、阿比纳夫·乔里、阿比纳夫·潘迪、阿布舍克·卡迪安、艾哈迈德·阿尔-达赫勒、艾莎·莱特曼、阿基尔·马图尔、艾伦·谢尔滕、亚历克斯·沃恩、艾米·杨、安吉拉·范、阿尼鲁德·戈亚尔、安东尼·哈特肖恩、奥博·杨、阿尔奇·米特拉、阿奇·斯劳万库马尔、阿尔テム·科列涅夫、亚瑟·欣斯瓦克、阿伦·拉奥、阿斯顿·张、奥雷利安·罗德里格斯、奥斯滕·格雷格森、艾娃·斯帕塔鲁、巴蒂斯特·罗齐耶尔、贝丝妮·比隆、宾·唐、鲍比·切恩、夏洛特·考谢特、查雅·纳亚克、克洛伊·比、克里斯·马拉、克里斯·麦康奈尔、克里斯蒂安·凯勒、克里斯托夫·图雷特、春阳·吴、科琳·黄、克里斯蒂安·坎通·费雷尔、赛勒斯·尼古拉迪斯、达米安·阿隆修斯、丹尼尔·宋、丹妮尔·品茨、丹尼·利夫希茨、丹尼·怀亚特、大卫·埃西奥布、德鲁夫·乔杜里、德鲁夫·马哈詹、迭戈·加西亚-奥拉诺、迭戈·佩里诺、迪乌克·胡普克斯、叶戈尔·拉科姆金、埃哈布·阿尔巴达维、埃琳娜·洛巴诺娃、艾米丽·迪南、埃里克·迈克尔·史密斯、菲利普·拉德诺维奇、弗朗西斯科·古兹曼、弗兰克·张、加布里埃尔·辛纳夫、加布里埃尔·李、乔治亚·刘易斯·安德森、戈文德·塔塔伊、格雷姆·奈尔、格雷瓜尔米亚隆, 关鹏, 吉列姆·库库雷尔, 海莉·阮, 汉娜·科雷瓦尔, 胡旭, 雨果·图夫龙, 伊利扬·扎罗夫, 伊马诺尔·阿列塔·伊巴拉, 伊莎贝尔·克劳曼, 伊尚·米斯拉, 伊万·埃夫季莫夫, 杰克·张, 杰德·科佩特, 在元·李, 扬·格费特, 雅娜·弗兰斯, 杰森·朴, 杰·马哈德奥卡, 吉特·沙阿, 杰尔默·范德林德, 詹妮弗·比洛克, 珍妮·洪, 叶夫根尼·李, 杰里米·傅, 建峰·池, 建宇

黄嘉文 刘杰 王杰草 余乔安娜 比顿·乔 斯皮萨克·钟洙 朴·约瑟夫 罗卡·约书亚 约翰斯顿·约书亚 萨克斯·俊腾 贾·卡利安 瓦苏登·阿尔瓦拉·卡尔蒂克 普拉萨德·卡尔蒂克亚 乌帕萨尼·凯特 普拉维亚克·科 李·肯尼思 希菲尔德·凯文 斯通·哈立德 埃尔-阿里尼·克里西卡 艾耶尔·克希蒂兹 马利克·奎恩利 邱·库纳尔 巴拉·库沙尔 拉霍蒂亚·劳伦 兰塔拉-耶里·劳伦斯 范德马滕·劳伦斯 陈亮 谭·丽兹 詹金斯·路易斯 马丁·洛维什 马丹·卢博 马洛·卢卡斯 布莱彻·卢卡斯 兰德扎特·卢克 德奥利维拉·玛德琳 穆齐·马赫什 帕苏普莱蒂·曼纳特 辛格·马诺哈尔 帕卢里·马尔钦 卡尔达斯·玛丽亚 齐姆普凯利·马修 奥尔德姆·马修 里塔·玛雅 帕夫洛娃·梅兰妮 坎巴杜尔·迈克 刘易斯·敏 西·米特什 库马尔·辛格·莫娜 哈桑·纳曼 戈亚尔·纳尔吉斯 托拉比·尼古拉 巴什利科夫·尼古拉 博戈伊切夫·尼拉德里 查特吉·宁 张·奥利维尔 杜申·奥努尔 切莱比·帕特里克 阿尔拉西·彭川 张鹏伟 李·佩塔尔 瓦西奇·彼得 翁·普拉吉瓦尔 巴尔加瓦·普拉蒂克 杜巴尔·普拉文 克里希南·普尼特 辛格·库拉·普克斯在 Xu, 清河, 董庆晓, 拉加万·斯里尼瓦桑, 拉杰·加纳帕西, 拉蒙·卡尔德勒, 里卡多·席尔维拉·卡布拉尔, 罗伯特·斯托伊奇尼克, 罗伯塔·莱莱努, 罗汉·马赫什瓦里, 罗希特·吉尔达尔, 罗希特·帕特尔, 罗曼·索维斯特, 罗尼·波利多罗, 罗珊·桑巴利, 罗斯·泰勒, 阮·席尔瓦, 侯锐, 王锐, 萨加尔·侯赛尼, 萨哈纳·切纳巴萨帕, 桑杰·辛格, 肖恩·贝尔, 徐贤·索尼娅·金, 谢尔盖·埃杜诺夫, 聂少良, Sharan Narang, 沙拉思·拉帕西, 盛申, 万圣业, 斯鲁蒂·布萨莱, 张顺, 西蒙·范登亨德, 索姆亚·巴特拉, 斯宾塞·惠特曼, 斯滕·苏特拉, 斯特凡·科洛特, 苏钦·古鲁兰甘, 悉尼·博罗丁斯基, 塔玛尔·赫尔曼, 塔拉·福勒, 塔里克·谢沙, 托马斯·乔治乌, 托马斯·西亚洛姆, 托比亚斯·斯佩克巴赫, 托多尔·米哈伊洛夫, 童晓, 乌贾瓦尔·卡恩, 韦达努吉·戈斯瓦米, 维布霍尔·古普塔, 维格内什·拉马纳森, 维克多·克尔克兹, 文森特·贡古埃, 维尔吉妮·多, 维什·沃格蒂, 维托尔·阿尔比耶罗, 弗拉丹·彼得罗维奇, 楚伟伟, 熊文瀚, 傅文印, 惠特尼·米尔斯, 泽维尔·马蒂内, 王晓东, 王晓芳, 谭晓庆·艾伦, 夏希德, 谢新峰, 徐查 的论文中贾学伟 王亚埃勒 戈尔德施拉格,亚什·高尔,亚斯明·巴巴埃伊 易文易文 宋宇晨 张悦 李宇宁 毛扎卡里 德尔皮埃尔·库德尔特,郑岩 郑兴·陈 佐伊·帕帕基波斯 阿迪亚·辛格,阿尤什·斯里瓦斯塔瓦 阿芭·贾恩 亚当·凯尔西 亚当·沙因费尔德 阿迪亚·甘迪迪 阿道夫·维多利亚,阿胡瓦·戈尔德斯坦德 阿贾伊·梅农 阿贾伊·夏尔马 亚历克斯·博森伯格 阿列克谢·巴耶夫斯基 艾莉·范斯坦 阿米特·桑加尼 阿莫斯·特奥 阿纳姆·尤努斯 安德烈·卢普 安德烈斯·阿尔瓦拉多 安德鲁·卡普尔斯 安德鲁·顾 安德鲁·波尔顿 安德鲁·瑞安 安基特·拉姆昌达尼 安妮·董 安妮·弗朗哥 阿努吉·戈亚尔 阿帕拉吉塔·萨拉夫 阿尔卡班杜·乔杜里 阿什利·加布里埃尔 阿什温·巴拉姆贝 阿萨夫·艾森曼 阿扎德·亚兹丹 博·詹姆斯 本·莫瑞尔 本杰明·莱昂哈迪 伯尼·黄 贝丝·劳埃德 贝托·德·保拉 巴尔加维·帕兰加佩 刘冰 吴波 博宇倪 布雷登·汉考克 布拉姆·瓦斯蒂 布兰登·斯宾塞 布拉尼·斯托伊科维奇 布莱恩·加米多 布里特·蒙塔尔沃 卡尔·帕克 卡莉·伯顿 卡塔利娜·梅希亚 刘策 王昌汉 金昌奎 周超 切斯特·胡 朱庆祥 克里斯·蔡 克里斯·廷达尔 克里斯托夫·费希滕霍费尔 辛西娅·高 达蒙·西文 达娜比蒂·丹尼尔·克雷默 丹尼尔·李 大卫·阿德金斯 大卫·徐 大卫德·泰斯图吉内 德利娅·大卫 黛维·帕里赫 戴安娜·利斯科维奇 迪德姆·福斯 丁康·王 杜克·勒,达斯汀·荷兰 爱德华·唐林 艾萨·贾米尔 伊莱恩·蒙哥马利 埃莉诺拉·普雷萨尼 艾米丽·哈恩 艾米丽·伍德 埃里克-图安·勒 埃里克·布林克曼 埃斯特班·阿尔考特 埃文·邓巴 埃文·斯莫瑟斯 飞·孙 费利克斯·克鲁克 峰·田 菲利波斯·科基诺斯 菲拉特·奥兹格内尔 弗朗切斯科·卡乔尼 弗兰克·卡纳耶特 弗兰克·赛德 加布里埃拉·梅迪纳·弗洛雷斯,加布里埃拉·施瓦茨 加达·巴迪尔 乔治亚·斯威 吉尔·哈尔珀恩 格兰特·赫尔曼 格里戈里·西佐夫 光义·张古纳·拉克什米纳拉亚南 哈坎·伊南 哈米德·肖贾纳泽里 汉·邹 汉娜·王汉文·赵 哈伦·哈比卜 哈里森·鲁道夫 海伦·苏克 亨利·阿斯佩格伦 亨特·戈德曼 宏远·展 易卜拉欣·达姆拉吉 伊戈尔·莫利博格 伊戈尔·图法诺夫 伊利亚斯·莱昂蒂亚迪斯 伊琳娜-埃琳娜·韦利切 伊泰·加特 杰克·韦斯曼詹姆斯·格博斯基 詹姆斯·科利 珍妮丝·兰 迦弗·亚设 让-巴蒂斯特·盖亚 杰夫·马库斯杰夫·唐 詹妮弗·陈 珍妮·甄 杰里米·赖岑斯坦 杰里米·特布尔 杰西卡·钟 建·金 静怡·杨 乔·卡明斯 J卡维尔·乔恩·谢泼德 乔纳森·麦克菲 乔纳森·托雷斯 乔什·金斯伯格 王俊杰 吴凯 坎·侯·U 卡兰·萨克塞纳 卡蒂凯·坎德尔瓦尔 卡塔尤恩·赞德 凯西·马托西奇 考希克·维拉拉加万 凯利·米歇尔纳 李克谦 基兰·贾加迪什 黄坤 库纳尔·乔拉 凯尔·黄 陈来林 拉克沙·加格 薰衣草A 莱昂德罗·席尔瓦 李·贝尔 张磊 郭良鹏 于立诚 利隆·莫什科维奇 卢卡·韦尔施泰特 马迪安·哈卜萨 马纳夫·阿瓦拉尼 马尼什·巴特 马丁纳斯·曼库斯 马坦·哈森 马修·莱尼 马蒂亚斯·雷索 马克西姆·格罗舍夫 马克西姆·瑙莫夫 玛雅·拉西 梅根·肯尼利 刘淼 迈克尔·L·塞尔泽 米哈尔·瓦尔科 米歇尔·雷斯特雷波 米希尔·帕特尔 米克·维亚特斯科夫 米卡埃尔·萨姆维良 迈克·克拉克 迈克·梅西 王迈克 米克尔·胡贝特·埃尔莫索 莫·梅塔纳特 穆罕默德

马德·拉斯特加里, 穆尼什·班萨尔, 南迪尼·桑塔纳姆, 娜塔莎·帕克斯, 娜塔莎·怀特, 纳夫亚塔·巴瓦, 纳扬·辛格尔, 尼克·埃格博, 尼古拉斯·尤苏尼尔, 尼基尔·梅塔, 尼古拉·巴甫洛维奇·拉普捷夫, 宁东, 诺曼·程, 奥列格·切尔诺古兹, 奥利维亚·哈特, 奥姆卡尔·萨尔佩卡尔, 厄兹莱姆·卡林利, 帕金·肯特, 帕斯·帕雷克, 保罗·萨阿布, 帕万·巴拉吉, 佩德罗·里特纳, 菲利普·邦特拉格, 皮埃尔·鲁, 皮奥特尔·多拉尔, 波琳娜·兹维亚吉娜, 普拉尚特·拉坦昌达尼, 普里蒂什·尤夫拉吉, 钱亮, 拉沙德·阿拉奥, 瑞秋·罗德里格斯, 拉菲·阿尤布, 拉格托汉·穆尔蒂, 拉古·纳亚尼, 拉胡尔·米特拉, 兰加普拉布·帕塔萨拉西, 雷蒙德·李, 丽贝卡·霍根, 罗宾·巴蒂, 洛奇·王, 拉斯·豪斯, 鲁迪·里诺特, 萨钦·梅塔, 萨钦·西比, 赛·贾耶什·邦杜, 萨米亚克·达塔, 萨拉·丘格, 萨拉·亨特, 萨尔贡·迪隆, 萨沙·西多罗夫, 萨塔德鲁·潘, 索拉布·马哈詹, 索拉布·维尔马, 清治·山本, 沙拉德·拉马斯瓦米, 肖恩·林赛, 肖恩·林赛, 盛峰, 盛浩·林, 盛欣·辛迪·赵, 希希尔·帕蒂尔, 希瓦·香卡尔, 书强·张, 书强·张, 思农·王, 斯内哈·阿格拉瓦尔, 索吉萨朱伊格贝、苏密特·钦塔拉、斯蒂芬妮·马克思、斯蒂芬·陈、史蒂夫·基霍、史蒂夫·萨特菲尔德、苏达申·戈文达普拉萨德、苏米特·古普塔、萨默·邓、成民·赵、桑尼·维尔克、苏拉杰·苏布拉马尼安、赛·乔杜里、悉尼·戈德曼、塔尔·雷梅兹、塔玛尔·格拉泽、塔玛拉·贝斯特、蒂洛·科勒、托马斯·罗宾逊、李天河、张天俊、蒂姆·马修斯、蒂莫西·周、祖克·沙凯德、瓦伦·冯蒂米塔、维多利亚·阿贾伊、维多利亚·蒙塔内兹、维杰·莫汉、维奈·萨蒂什·库马尔、维沙尔·曼格拉、弗拉德·伊万内斯库、弗拉德·波埃纳鲁、弗拉德·蒂贝里乌·米哈莱斯库、弗拉基米尔·伊万诺夫、李伟、王文臣、蒋文文、韦斯·布阿齐兹、威尔·康斯特布尔、唐晓成、吴晓健、王晓兰、吴希伦、高新波、亚尼夫·克莱因曼、陈延军、胡叶、贾叶、齐叶、李燕达、张一林、张颖、约西·阿迪、永镇·南、余、王、赵宇、郝宇辰、钱韵迪、李云露、何雨子、扎克·赖特、扎卡里·德维托、泽夫·罗森布里克、文兆多、杨振宇、赵志伟和马志宇。The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.

安阳, 宝松杨, 北辰张, 宾远惠, 博政, 博文宇, 成远李, 大一恒刘, 飞黄, 浩然魏, 欢林, 建杨, 建红图, 建伟张, 建新杨, 嘉喜杨, 敬人周, 俊阳林, 凯当, 克明陆, 克勤鲍, 科信杨, 乐宇, 梅李, 明峰薛, 佩张, 琴朱, 瑞门, 润吉林, 天浩李, 廷宇夏, 兴章任, 宣成任, 杨帆, 杨苏, 益昌张, 宇万, 玉琼刘, 泽宇崔, 真如张, 和子涵邱. Qwen2.5 技术报告. arXiv preprint arXiv:2412.15115, 2024.  
Aakanksha Chowdhery, Sharan Narang, 雅各布·德夫林, Maarten Bosma, Gaurav Mishra, 亚当·罗伯茨, 保罗·巴勒姆, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, 诺姆·沙泽尔, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, 詹姆斯·布拉德伯里, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, 桑杰·格马瓦特, Sunipa Dev, Henryk Michalewski, 哈维尔·加西亚, Vedant Misra, 凯文·罗宾逊, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, 雷翁·柴尔德, Oleksandr Polozov, 凯瑟琳·李, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, 和 Noah Fiedel。PaLM: Scaling language modeling with pathways, 2022. arXiv:2204.02311。
Dan Hendrycks 和 Kevin Gimpel。Bridging nonlinearities and stochastic regularizers with gaussian error linear units, 2016. arXiv:1606.08415。
Stefan Elfwing, Eiji Uchibe, 和 Kenji Doya。Sigmoid-weighted linear units for neural network function approximation in reinforcement learning, 2017. URL https://arxiv.org/abs/1702.03118。
Yann N. Dauphin, Angela Fan, Michael Auli, 和 David Grangier。Language modeling with gated convolutional networks, 2017. URL https://arxiv.org/abs/1612.08083。
Noam Shazeer。GLU variants improve transformer, 2020. arXiv:2002.05202。
Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, 和 Yunfeng Liu。Roformer: Enhanced transformer with rotary position embedding, 2021。

Diederik P. Kingma 和 Jimmy Ba。Adam: A method for stochastic optimization. In Proc. of ICLR, 2015。
Ilya Loshchilov 和 Frank Hutter。Decoupled weight decay regularization. In Proc. of ICLR, 2019。  
汤姆·B·布朗、本杰明·曼恩、尼克·莱德、梅兰妮·苏比亚、贾里德·卡普兰、普拉富拉·达里瓦尔、阿尔温德·尼拉克坦坦、普拉纳夫·夏姆、吉里什·萨斯特里、阿曼达·阿斯凯尔、桑迪尼·阿格拉瓦尔、阿里尔·赫伯特-沃斯、格蕾琴·克鲁格、汤姆·亨尼根、雷翁·柴尔德、阿迪亚·拉梅什、丹尼尔·M·齐格勒、杰弗里·吴、克莱门斯·温特、克里斯托弗·赫塞、马克·陈、埃里克·西格勒、马特乌什·利特温、斯科特·格雷、本杰明·切斯、杰克·克拉克、克里斯托弗·伯纳、萨姆·麦坎德利什、亚历克·拉德福德、伊利亚·苏茨克维和达里奥·阿莫代。Language models are few-shot learners. In Proc. of NeurIPS, 2020.
贾里德·卡普兰、萨姆·麦坎德利什、汤姆·亨尼根、汤姆·B·布朗、本杰明·切斯、雷翁·柴尔德、斯科特·格雷、亚历克·拉德福德、杰弗里·吴和达里奥·阿莫代。Scaling laws for neural language models, 2020. arXiv:2001.08361.  
乔丹·霍夫曼，塞巴斯蒂安·博尔若，亚瑟·门施，埃琳娜·布查茨卡娅，特雷弗·蔡，伊丽莎·卢瑟福，迭戈·德·拉斯·卡萨斯，丽莎·安妮·亨德里克斯，约翰内斯·韦尔布尔，艾丹·克拉克，汤姆·亨尼根，埃里克·诺兰德，凯蒂·米利坎，乔治·范登德里谢，博格丹·达莫克，奥蕾莉亚·盖伊，西蒙·奥辛德罗，凯伦·西蒙扬，埃里希·埃尔森，杰克·W·雷，奥里奥尔·维尼亚尔斯，和洛朗·西弗雷。Training compute-optimal large language models, 2022. arXiv:2203.15556.
阿里·霍尔茨曼，扬·拜斯，李杜，麦克斯韦尔·福布斯，和崔艺珍。The curious case of neural text degeneration. In Proc. of ICLR, 2020.  
姚宏胡伯特蔡、邵杰白、山田诚、路易-菲利普·莫伦西和鲁斯兰·萨拉赫特迪诺夫。Transformer dissection: An unified understanding for transformer's attention via the lens of kernel。在 Kentaro Inui、Jing Jiang、Vincent Ng 和 Xiaojun Wan 编辑的《Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)》中，第 4344-4353 页，香港，中国，2019 年 11 月。Association for Computational Linguistics。doi: 10.18653/v1/D19-1443。URL https://aclanthology.org/D19-1443/。
Amirhossein Kazemnejad、Inkit Padhi、Karthikeyan Natesan、Payel Das 和 Siva Reddy。The impact of positional encoding on length generalization in transformers。在《Thirty-seventh Conference on Neural Information Processing Systems》中，2023 年。URL https://openreview.net/forum?id=Drrl2gcjzl。