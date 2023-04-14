from mindspore import load_checkpoint, load_param_into_net

from mindspore import nn

from src.train_and_evaluate import *
from src.models import *
from src.lr_schedule import dynamic_lr
from src.expressions_transfer import *
from src.untitled import *
import time
from mindspore import context

from src.expressions_transfer import *

batch_size = 250
embedding_size = 128
hidden_size = 512
n_epochs = 20
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2

context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')

data_all = load_raw_data("/root/seq2tree/data/Math_23K_interpretable.json") # 6666 instances
data_train = load_raw_data("/root/seq2tree/data/Math_23K_train.json") # 5332 instances
data_test = load_raw_data("/root/seq2tree/data/Math_23K_test.json") # 1334 instances

_, generate_nums, copy_nums = transfer_num(data_all)
pairs_train, _, _ = transfer_num(data_train)
pairs_test, _, _ = transfer_num(data_test)


temp_pairs = []
for p in pairs_train:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs_train = temp_pairs

temp_pairs = []
for p in pairs_test:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs_test = temp_pairs

pairs_trained = pairs_train
pairs_tested = pairs_test
input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
                                                                copy_nums, tree=True)

# 将模型参数存入parameter的字典中，这里加载的是上面训练过程中保存的模型参数
param_dict = load_checkpoint("MyNet.ckpt")

encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                     n_layers=n_layers)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                     input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)

withloss = Seq2TreeWithLossCell(encoder, predict, generate, merge, output_lang.word2index["UNK"], output_lang.num_start)

warmup_step = 300
warmup_ratio = 0.33
base_step = len(data_all)//batch_size + 1
lr_dynamic = dynamic_lr(learning_rate, n_epochs, warmup_step, warmup_ratio, base_step)

optimizer = nn.Adam(withloss.trainable_params(), learning_rate=lr_dynamic, weight_decay=weight_decay)

trainonestep = S2TTrainOneStepCell(withloss, optimizer)

# 将参数加载到网络中
load_param_into_net(trainonestep, param_dict)

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])
    
value_ac = 0
equation_ac = 0
eval_total = 0
start = time.time()
index = 0
for test_batch in test_pairs:
    test_res = evaluate_tree(trainonestep.network.encoder, trainonestep.network.predict, trainonestep.network.generate, trainonestep.network.merge, test_batch[0], test_batch[1], generate_num_ids, test_batch[5], output_lang.num_start)
    print(index)
    index += 1
    val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
    if val_ac:
        value_ac += 1
    if equ_ac:
        equation_ac += 1
    eval_total += 1
print(equation_ac, value_ac, eval_total)
print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
print("testing time", time_since(time.time() - start))
print("------------------------------------------------------")